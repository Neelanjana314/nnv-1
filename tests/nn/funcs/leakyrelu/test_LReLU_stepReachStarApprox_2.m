lb = [-0.5; -0.5];
ub = [0.5; 0.5];

B = Box(lb, ub);
I1 = B.toZono;

A = [0.5 1; 1.5 -2];
I = I1.affineMap(A, []);
S = I.toStar;

figure;
S.plot;
R = PosLin.stepReachStarApprox(S, 1);
figure;
R.plot;