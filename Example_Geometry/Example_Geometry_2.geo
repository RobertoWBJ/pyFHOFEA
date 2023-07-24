// Gmsh project created on Sat Jul 15 14:35:15 2023
//+
Point(1) = {0, 0, 0, 2};
//+
Point(2) = {0, 150, 0, 2};
//+
Point(3) = {30, 150, 0, 2};
//+
Point(4) = {30, 0, 0, 2};
//+
Point(5) = {80, 50, 0, 2};
//+
Point(6) = {80, 125, 0, 2};
//+
Point(7) = {300, 125, 0, 2};
//+
Point(8) = {300, 50, 0, 2};
//+
Point(9) = {55, 137.5, 0, 2};
//+
Point(10) = {55, 25, 0, 2};
//+
Point(11) = {105, 50, 0, 2};
//+
Point(12) = {105, 125, 0, 2};
//+
Point(13) = {150, 87.5, 0, 2};
//+
Point(14) = {255, 87.5, 0, 2};
//+
Point(15) = {255, 102.5, 0, 2};
//+
Point(16) = {255, 72.5, 0, 2};
//+
Point(17) = {150, 72.5, 0, 2};
//+
Point(18) = {150, 102.5, 0, 2};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 9};
//+
Line(4) = {9, 6};
//+
Line(5) = {6, 12};
//+
Line(6) = {12, 7};
//+
Line(7) = {7, 8};
//+
Line(8) = {8, 11};
//+
Line(9) = {11, 5};
//+
Line(10) = {5, 10};
//+
Line(11) = {10, 4};
//+
Line(12) = {4, 1};
//+
Circle(13) = {18, 13, 17};
//+
Circle(14) = {16, 14, 15};
//+
Line(15) = {18, 15};
//+
Line(16) = {16, 17};
//+
Line(17) = {12, 11};
//+
Line(18) = {10, 9};
//+
Curve Loop(1) = {1, 2, 3, -18, 11, 12};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {18, 4, 5, 17, 9, 10};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {6, 7, 8, -17};
//+
Curve Loop(4) = {13, -16, 14, -15};
//+
Plane Surface(3) = {3, 4};
//+
Physical Curve("LE", 19) = {1};
//+
Physical Curve("RE", 20) = {7};
//+
Physical Curve("TE", 21) = {5, 6};
//+
Physical Curve("BE", 22) = {9, 8};
//+
Physical Curve("IBE", 23) = {10, 11};
//+
Physical Curve("ITE", 24) = {3, 4};
//+
Physical Surface("S1", 25) = {1};
//+
Physical Surface("S2", 26) = {2};
//+
Physical Surface("S3", 27) = {3};
