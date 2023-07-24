// Gmsh project created on Tue Jul 04 22:22:00 2023
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0, 45, 0, 1.0};
//+
Point(3) = {150, 45, 0, 1.0};
//+
Point(4) = {150, 0, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Point(5) = {75, 22.5, 0, 1.0};
//+
Point(6) = {86.25, 22.5, 0, 1.0};
//+
Point(7) = {63.75, 22.5, 0, 1.0};
//+
Circle(5) = {7, 5, 6};
//+
Circle(6) = {6, 5, 7};
//+
Curve Loop(1) = {2, 3, 4, 1};
//+
Curve Loop(2) = {6, 5};
//+
Plane Surface(1) = {1, 2};
//+
Physical Curve("LE", 7) = {1};
//+
Physical Curve("RE", 8) = {3};
//+
Physical Curve("TE", 9) = {2};
//+
Physical Curve("BE", 10) = {4};
//+
Physical Surface("S1", 11) = {1};
