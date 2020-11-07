
I = ExamplePoly.randVrep;   
V = [0 0; 1 1; 0.5 1];
I = Star(V', I.A, I.b); % input star
X = I.sample(100);

figure;
I.plot;
hold on;
plot(X(1, :), X(2, :), 'ob'); % sampled inputs

S1 = LReLU.reach(I); % exach reach set
tic;
S2 = LReLU.reach_star_approx(I); 
toc;
% tic;
% S3 = PosLin.reach_star_approx_fast(I); % fast - over-approximate reach set
% toc;
alpha = 0.01;
Y = LReLU.evaluate(X, alpha);

figure;
% S3.plot; 
% hold on;
S2.plot;
hold on;
Star.plots(S1);
hold on;
plot(Y(1, :), Y(2, :), '*'); % sampled outputs
