import csv
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import meshio

class Pre_processing():

    def __init__(self, mesh_path: str, DOF: int = 2):
        self.mesh_path = mesh_path
        mesh = meshio.read(mesh_path)
        self.point = mesh.points  # importa os nós
        self.cells = mesh.cells_dict  # importa os elementos
        self.cell_data = mesh.cell_data_dict  # importa a relação entre a geometria e o physical group
        self.physical_group = mesh.field_data  # importa os nomes e as referências do physical group

        x = self.point[:, 0]
        y = self.point[:, 1]
        elements = self.cells['triangle']
        self.triangulation = tri.Triangulation(x, y, elements)

        self.DOF_structure = len(self.point) * DOF  # verifica o Nº de GDLs da estrutura

        # estabelece a matriz coluna de força
        self.force_matrix = np.zeros(self.DOF_structure)

        self.restrictions = np.array([], dtype='int32')  # inicializa a matriz de restrições

        self.thickness = 0
        self.young_modulus = 0
        self.poisson_coef = 0

        pass

    def export2APDL(self, file_name: str):
        txt_name = self.mesh_path[:-4]

        with open(txt_name + '.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['/prep7'])
            writer.writerow(['real', 1])
            writer.writerow(['et', 1, 'PLANE182'])
            writer.writerow(['type', 1])
            writer.writerow(['mat', 1])

            for pos, point in enumerate(self.point):
                row = ['n', pos + 1, point[0], point[1], point[2]]
                writer.writerow(row)

            for pos, node in enumerate(self.cells['triangle']):
                node = node + 1
                row = ['e', node[0], node[1], node[2]]
                writer.writerow(row)

            writer.writerow(['cdwrite', 'db', file_name, 'cdb'])

        pass

    def get_nodes_from_physical_group(self, bc: str, output_index=False):
        geom_db = {0: 'vertex', 1: 'line', 2: 'triangle'}
        physical_tag = self.physical_group[bc][0]
        geom_type = geom_db[self.physical_group[bc][1]]
        physical_tag_position = np.where(self.cell_data['gmsh:physical'][geom_type] == physical_tag)[0]
        first_index = physical_tag_position.min()
        last_index = physical_tag_position.max() + 1

        element_reference = self.cells[geom_type][first_index:last_index]

        if output_index == False:
            return geom_type, element_reference
        else:
            return geom_type, element_reference, first_index, last_index

    def plot_geometry(self):
        plt.figure(figsize=(max(self.triangulation.x) / 10, max(self.triangulation.y) / 10))
        plt.axis('equal')
        plt.axis('off')
        plt.triplot(self.triangulation, color='black')
        plt.show()

        pass

    def apply_forces(self, bc: str, Fx: float = 0, Fy: float = 0):
        geom_type, element_reference = Pre_processing.get_nodes_from_physical_group(self, bc)

        if geom_type == 'vertex':
            unique_nodes_id_list = np.unique(np.array(element_reference).flatten())
            N_nodes = len(unique_nodes_id_list)
            Fx = Fx / N_nodes
            Fy = Fy / N_nodes
            for node_id in unique_nodes_id_list:
                self.force_matrix[2 * node_id] += Fx
                self.force_matrix[2 * node_id + 1] += Fy

        elif geom_type == 'line':
            length_list = np.array([], dtype='float64')

            for line in element_reference:
                x0, y0 = self.point[line[0]][0], self.point[line[0]][1]
                x1, y1 = self.point[line[1]][0], self.point[line[1]][1]
                length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                length_list = np.append(length_list, length)

            total_length = length_list.sum()

            Fx_m = Fx / (2 * total_length)
            Fy_m = Fy / (2 * total_length)

            for i, length in enumerate(length_list):
                node_id1 = element_reference[i][0]
                node_id2 = element_reference[i][1]

                Fx_ = Fx_m * length
                Fy_ = Fy_m * length

                self.force_matrix[2 * node_id1] += Fx_
                self.force_matrix[2 * node_id2] += Fx_
                self.force_matrix[2 * node_id1 + 1] += Fy_
                self.force_matrix[2 * node_id2 + 1] += Fy_

        else:
            raise Exception('Could not handle geometry type. Change for VERTEX or LINE.')

        pass

    def set_restrictions(self, bc: str, Rx: int = 1, Ry: int = 1):
        '''
        Rx (int) = 0 to constrain or 1 to maintain DOF
        Ry (int) = 0 to constrain or 1 to maintain DOF
        '''

        cond = 0 <= Rx <= 1 and 0 <= Ry <= 1

        if cond == False:
            raise Exception('''
            INVALID RESTRICTION

            Type 0 to restrict the DOF
            Type 1 to maintain the DOF

            ''')

        geom_type, element_reference = Pre_processing.get_nodes_from_physical_group(self, bc)

        unique_nodes_id_list = np.unique(np.array(element_reference).flatten())

        for node in unique_nodes_id_list:
            DOFx = 2 * node
            DOFy = 2 * node + 1
            if Rx == 0 and DOFx not in self.restrictions:
                self.restrictions = np.append(self.restrictions, DOFx)
            if Ry == 0 and DOFy not in self.restrictions:
                self.restrictions = np.append(self.restrictions, DOFy)

        pass

    def set_physical_properties(self, thickness: float, young_modulus: float, poisson_coef: float):
        self.thickness = thickness
        self.young_modulus = young_modulus
        self.poisson_coef = poisson_coef


class Processing():

    def __init__(self, inputs: Pre_processing):
        self.elements = inputs.cells['triangle']
        self.inputs = inputs

        self.displacement_matrix = np.zeros(self.inputs.DOF_structure)
        self.K_matrix = np.zeros((self.inputs.DOF_structure, self.inputs.DOF_structure))
        self.stress_matrix = np.zeros((len(self.elements), 9))
        self.displaced_point = np.copy(self.inputs.point[:, :2])

    def K_matrix_assemblage(self):
        C1 = 1  # constant C1 used in the element stiffness matrix (plane stress case)
        C2 = 0.5 * (1 - self.inputs.poisson_coef)  # constant C2 used in the element stiffness matrix (plane stress case)

        for nodes in self.elements:
            xi, yi = self.inputs.point[nodes[0]][:2]
            xj, yj = self.inputs.point[nodes[1]][:2]
            xm, ym = self.inputs.point[nodes[2]][:2]

            A = abs(0.5 * (xi * (yj - ym) + xj * (ym - yi) + xm * (yi - yj)))  # element area

            # constant part of the element stiffness matrix
            k_const = self.inputs.thickness * self.inputs.young_modulus / (4 * A * (1 - self.inputs.poisson_coef ** 2))

            bi = yj - ym  # betha_i factor
            bj = ym - yi  # betha_j factor
            bm = yi - yj  # betha_m factor
            gi = xm - xj  # gamma_i factor
            gj = xi - xm  # gamma_j factor
            gm = xj - xi  # gamma_m factor

            k11 = bi ** 2 * C1 + gi ** 2 * C2
            k12 = k21 = bi * gi * self.inputs.poisson_coef + bi * gi * C2
            k13 = k31 = bi * bj * C1 + gi * gj * C2
            k14 = k41 = bi * gj * self.inputs.poisson_coef + bj * gi * C2
            k15 = k51 = bi * bm * C1 + gi * gm * C2
            k16 = k61 = bi * gm * self.inputs.poisson_coef + bm * gi * C2
            k22 = gi ** 2 * C1 + bi ** 2 * C2
            k23 = k32 = bj * gi * self.inputs.poisson_coef + bi * gj * C2
            k24 = k42 = gi * gj * C1 + bi * bj * C2
            k25 = k52 = bm * gi * self.inputs.poisson_coef + bi * gm * C2
            k26 = k62 = gi * gm * C1 + bi * bm * C2
            k33 = bj ** 2 * C1 + gj ** 2 * C2
            k34 = k43 = bj * gj * self.inputs.poisson_coef + bj * gj * C2
            k35 = k53 = bj * bm * C1 + gj * gm * C2
            k36 = k63 = bj * gm * self.inputs.poisson_coef + gj * bm * C2
            k44 = gj ** 2 * C1 + bj ** 2 * C2
            k45 = k54 = bm * gj * self.inputs.poisson_coef + bj * gm * C2
            k46 = k64 = gj * gm * C1 + bj * bm * C2
            k55 = bm ** 2 * C1 + gm ** 2 * C2
            k56 = k65 = gm * bm * self.inputs.poisson_coef + bm * gm * C2
            k66 = gm ** 2 * C1 + bm ** 2 * C2

            k_element = k_const * np.array([[k11, k12, k13, k14, k15, k16],
                                            [k21, k22, k23, k24, k25, k26],
                                            [k31, k32, k33, k34, k35, k36],
                                            [k41, k42, k43, k44, k45, k46],
                                            [k51, k52, k53, k54, k55, k56],
                                            [k61, k62, k63, k64, k65, k66]])

            # the matrix below correlates the DOF's of the active element
            # and the structure stiffness matrix
            DOF_correlation_list = np.array([nodes[0] * 2, nodes[0] * 2 + 1,
                                             nodes[1] * 2, nodes[1] * 2 + 1,
                                             nodes[2] * 2, nodes[2] * 2 + 1])

            for Ki, ki in zip(DOF_correlation_list, range(6)):
                for Kj, kj in zip(DOF_correlation_list, range(6)):
                    self.K_matrix[Ki, Kj] += k_element[ki, kj]

    def apply_restrictions(self):
        K_matrix_cut = np.copy(self.K_matrix)
        force_matrix_cut = np.copy(self.inputs.force_matrix)

        K_matrix_cut = np.delete(np.delete(K_matrix_cut, self.inputs.restrictions, 0), self.inputs.restrictions, 1)

        force_matrix_cut = np.delete(force_matrix_cut, self.inputs.restrictions, 0)

        return K_matrix_cut, force_matrix_cut

    def solve(self):
        K_matrix, force_matrix = Processing.apply_restrictions(self)

        it_displacement = iter(np.linalg.solve(K_matrix, force_matrix))

        for i in range(self.inputs.DOF_structure):
            if i not in self.inputs.restrictions:
                self.displacement_matrix[i] = next(it_displacement)

        pass

    def get_all_results(self):
        for i in range(len(self.displaced_point)):
            self.displaced_point[i, 0] += self.displacement_matrix[2 * i]
            self.displaced_point[i, 1] += self.displacement_matrix[2 * i + 1]

        C2 = 0.5 * (1 - self.inputs.poisson_coef)  # constant C2 used in the element stiffness matrix (plane stress case)

        for pos, nodes in enumerate(self.elements):
            xi, yi = self.inputs.point[nodes[0]][:2]
            xj, yj = self.inputs.point[nodes[1]][:2]
            xm, ym = self.inputs.point[nodes[2]][:2]

            d_matrix = np.array([self.displacement_matrix[2 * nodes[0]],
                                 self.displacement_matrix[2 * nodes[0] + 1],
                                 self.displacement_matrix[2 * nodes[1]],
                                 self.displacement_matrix[2 * nodes[1] + 1],
                                 self.displacement_matrix[2 * nodes[2]],
                                 self.displacement_matrix[2 * nodes[2] + 1], ])

            A = abs(0.5 * (xi * (yj - ym) + xj * (ym - yi) + xm * (yi - yj)))

            # constant part of the element stiffness matrix
            k_const = self.inputs.young_modulus / (2 * A * (1 - self.inputs.poisson_coef ** 2))

            bi = yj - ym  # betha_i factor
            bj = ym - yi  # betha_j factor
            bm = yi - yj  # betha_m factor
            gi = xm - xj  # gamma_i factor
            gj = xi - xm  # gamma_j factor
            gm = xj - xi  # gamma_m factor

            k11 = bi
            k12 = self.inputs.poisson_coef * gi
            k13 = bj
            k14 = self.inputs.poisson_coef * gj
            k15 = bm
            k16 = self.inputs.poisson_coef * gm
            k21 = self.inputs.poisson_coef * bi
            k22 = gi
            k23 = self.inputs.poisson_coef * bj
            k24 = gj
            k25 = self.inputs.poisson_coef * bm
            k26 = gm
            k31 = C2 * gi
            k32 = C2 * bi
            k33 = C2 * gj
            k34 = C2 * bj
            k35 = C2 * gm
            k36 = C2 * bm

            DB_matrix = np.array([[k11, k12, k13, k14, k15, k16],
                                  [k21, k22, k23, k24, k25, k26],
                                  [k31, k32, k33, k34, k35, k36]])

            sigma_X, sigma_Y, tau_XY = -k_const * DB_matrix @ d_matrix

            tau_max_plane = np.sqrt(((sigma_X - sigma_Y) / 2) ** 2 + tau_XY ** 2)
            sigma_max = (sigma_X + sigma_Y) / 2 + tau_max_plane
            sigma_min = (sigma_X + sigma_Y) / 2 - tau_max_plane

            strain_X = sigma_X / self.inputs.young_modulus
            strain_Y = sigma_Y / self.inputs.young_modulus

            sigma_von_mises = np.sqrt(sigma_max ** 2 - sigma_max * sigma_min + sigma_min ** 2)

            self.stress_matrix[pos, :] = np.array([strain_X, strain_Y,
                                                   sigma_X, sigma_Y, tau_XY,
                                                   sigma_max, sigma_min, tau_max_plane,
                                                   sigma_von_mises])

        pass


class Post_processing():

    def __init__(self, pre_data: Pre_processing, pro_data: Processing):
        self.x = pre_data.point[:, 0]
        self.y = pre_data.point[:, 1]
        self.point = pre_data.point
        self.plot_geometry = pre_data.plot_geometry
        self.get_nodes_from_physical_group = pre_data.get_nodes_from_physical_group
        self.displacement_matrix = pro_data.displacement_matrix
        self.stress_results = pro_data.stress_matrix
        self.elements = pre_data.cells['triangle']
        self.color_contour = ListedColormap(["#0000FF", "#00B2FF", "#00FFFF", "#00FFB2",
                                             "#00FF00", "#B2FF00", "#FFFF00", "#FFB200", "#FF0000"])
        self.triangulation = tri.Triangulation(self.x, self.y, self.elements)
        pass

    def global_plot_func(self, triangulation, contour, title, units, scale, triplot=False):
        if triplot == False:
            plt.figure(figsize=(max(triangulation.x) / (10), max(triangulation.y) / (10)))
            ticks = np.linspace(min(contour), max(contour), 10)
            plt.axis('equal')
            plt.axis('off')
            plt.title(title, fontsize=20)
        else:
            plt.figure(figsize=(max(self.triangulation.x) / (10), max(self.triangulation.y) / (10)))
            if np.std(contour) / np.mean(contour) <= 0.0001:
                contour = np.ones(len(contour)) * np.mean(contour)
                ticks = np.ones(1) * np.mean(contour)
            else:
                ticks = np.linspace(min(contour), max(contour), 10)
            plt.axis('equal')
            plt.axis('off')
            plt.title(title, fontsize=20)
            plt.triplot(self.triangulation, color='black', linewidth=0.1)
        tpc = plt.tripcolor(triangulation, facecolors=contour,
                            edgecolors='none', cmap=self.color_contour)
        cbar = plt.colorbar(tpc, ticks=ticks, format='%.03e')
        cbar.set_label(label=units, size=20)
        cbar.ax.tick_params(labelsize=20)
        plt.show()
        pass

    def local_plot_func(self, bc, contour, title, units, scale):
        geom_type, element_reference, first_index, last_index = self.get_nodes_from_physical_group(bc,
                                                                                                   output_index=True)
        contour = contour[first_index:last_index]

        base_matrix = np.zeros([len(element_reference) * 3, 4])

        if geom_type == 'triangle':
            for i, node in enumerate(element_reference):
                base_matrix[i * 3, 1:] = node[0], self.x[int(node[0])], self.y[int(node[0])]
                base_matrix[i * 3 + 1, 1:] = node[1], self.x[int(node[1])], self.y[int(node[1])]
                base_matrix[i * 3 + 2, 1:] = node[2], self.x[int(node[2])], self.y[int(node[2])]
        else:
            raise Exception('Could not handle element type. Change for TRIANGLE.')

        base_matrix = np.unique(base_matrix, axis=0)
        base_matrix = base_matrix[base_matrix[:, 1].argsort()]
        base_matrix[:, 0] = np.arange(len(base_matrix))
        for new_ref, old_ref in base_matrix[:, 0:2]:
            element_reference[element_reference == old_ref] = new_ref

        triangulation = tri.Triangulation(base_matrix[:, 2], base_matrix[:, 3], element_reference)
        Post_processing.global_plot_func(self, triangulation, contour, title,
                                         units, scale, triplot=True)
        pass

    def strain_X(self, units: str = ""):
        Post_processing.global_plot_func(self, self.triangulation, self.stress_results[:, 0],
                                         'Strain X', units, scale=1)
        pass

    def local_strain_X(self, bc: str, units: str = ""):
        Post_processing.local_plot_func(self, bc, self.stress_results[:, 0], 'Strain X', units, scale=1)

    def von_mises(self, units: str = ""):
        Post_processing.global_plot_func(self, self.triangulation, self.stress_results[:, 8],
                                         'von Mises', units, scale=1)
        pass

    def deformed_von_mises(self, scale: int = 1, units: str = ""):
        displaced_point = np.copy(self.point[:, :2])

        for i in range(len(displaced_point)):
            displaced_point[i, 0] += self.displacement_matrix[2 * i] * scale
            displaced_point[i, 1] += self.displacement_matrix[2 * i + 1] * scale

        x = displaced_point[:, 0]
        y = displaced_point[:, 1]
        self.displaced_triangulation = tri.Triangulation(x, y, self.elements)
        Post_processing.global_plot_func(self, self.displaced_triangulation, self.stress_results[:, 8],
                                         'von Mises', units, scale)
        pass