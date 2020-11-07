
I = ExamplePoly.randVrep;   
V = [0 0; 1 0; 0 1];
I = Star(V', I.A, I.b); % input star
X = I.sample(100);

figure;
I.plot;
hold on;
plot(X(1, :), X(2, :), 'ob'); % sampled inputs

S = LReLU.reach_abstract_domain(I); % over-approximate reach set
S1 = LReLU.reach_star_approx(I); % exach reach set
alpha = 0.01;
Y = LReLU.evaluate(X, alpha);

figure;
S.plot;
hold on;
Star.plots(S1);
hold on;
plot(Y(1, :), Y(2, :), '*'); % sampled outputs
