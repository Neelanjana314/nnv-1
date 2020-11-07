classdef LReLU
    % LeakyReLU operator in NN
    % Framework: Dung Tran
    % Modification: Neelanjana Pal
    
    properties
    end
    
    methods(Static)
        % evaluation
        function y=evaluate(x, alpha)
            [n, m] = size(x);
            y = x;
            for i=1:n
                for j=1:m
                    if x(i,j) < 0
                        y(i,j) = alpha*x(i,j);
                    end
                end
            end
            % map = find(y<0);
            % y(map) = alpha*x(map);
        end
        
        %step reach set y = LReLU(x)
        function S = stepReach(varargin)
            switch nargin
                case 2
                    I = varargin{1};
                    index = varargin{2};
                    lp_solver = 'linprog';
                case 3
                    I = varargin{1};
                    index = varargin{2};
                    lp_solver = varargin{3};
                otherwise
                    error('Invalid number of input arguments, should be 2 or 3');
            end
            alpha  = 0.01;
            if ~isa(I, 'Star')
                error('Input is not a star set');
            end
            xmin = I.getMin(index, lp_solver);
            xmax = I.getMax(index, lp_solver);
            
            if xmin >= 0
                S = I;
            elseif xmax <= 0
                %S = alpha*I;
                V = I.V;
                V(index, :) = alpha*V(index, :);
                
                if ~isempty(I.Z)
                    c1= I.Z.c;
                    %c1(index) = 0;
                    V1 = I.Z.V;
                    V1(index, :) = alpha*V1(index, :);
                    new_Z = Zono(c1, V1); % update outer-zono
                else
                    new_Z = [];
                end
                S = Star(V, I.C, I.d, I.predicate_lb, I.predicate_ub, new_Z);
            else
                c = I.V(index, 1);
                V = I.V(index, 2:I.nVar + 1);
                
                % S1 = I && x[index] < 0
                new_C = vertcat(I.C, V);
                new_d = vertcat(I.d, -c);
                new_V = I.V;
                new_V(index,:) = alpha*new_V(index,:);
                
                %update outer -zono
                if ~isempty(I.Z)
                    c1 = I.Z.c;
                    %c1(index) = alpha*c1(index);
                    V1 = I.Z.V;
                    V1(index,:) = alpha*V1(index,:);
                    new_Z = Zono(c1, V1);
                else
                    new_Z = [];
                end
                
                S1= Star(new_V, new_C, new_d, I.predicate_lb, I.predicate_ub, new_Z);
                
                % S2 = I && x[index] > 0
                new_C = vertcat(I.C, -V);
                new_d = vertcat(I.d, c);
                S2= Star(I.V, new_C, new_d, I.predicate_lb, I.predicate_ub, I.Z);
                
                S = [S1 S2];
            end
        end
        
%         function S = stepReach2(I, index)
%         end
        
        function S = stepReachMultipleInputs(varargin)
            
            switch nargin
                case 3
                    I = varargin{1};
                    index = varargin{2}; 
                    option = varargin{3};
                    lp_solver = 'linprog';
                case 4
                    I = varargin{1};
                    index = varargin{2}; 
                    option = varargin{3};
                    lp_solver = varargin{4};
                otherwise
                    error('Invalid number of input arguments');
            end
            %alpha = 0.01;
            p = length(I);
            S = [];
            
            if isempty(option)
                for i=1:p
                    S = [S, LReLU.stepReach(I(i), index, lp_solver)];
                end
            elseif strcmp(option, 'parallel')
                parfor i=1:p
                    S = [S, LReLU.stepReach(I(i), index, lp_solver)];
                end
            else
                error('Unknown option');
            end
        end
        
        function S = reach_star_exact(varargin)
            switch nargin
                case 2
                    I = varargin{1};
                    option = varargin{2};
                    dis_opt = [];
                    lp_solver = 'linprog';
                case 3
                    I = varargin{1};
                    option = varargin{2};
                    dis_opt = varargin{3};
                    lp_solver = 'linprog';
                case 4
                    I = varargin{1};
                    option = varargin{2};
                    dis_opt = varargin{3};
                    lp_solver = varargin{4};
                otherwise
                    error('Invalid number of input arguments');
            end
            alpha = 0.01;
            if ~isempty(I)
                [lb, ub] = I.estimateRanges;
                
                if isempty(lb) || isempty(ub)
                    S = [];
                else
                    map = find(ub <= 0); % computation map
                    V = I.V;
                    V(map, :) = alpha*V(map, :);
                    % update outer-zono
                    if ~isempty(I.Z)
                        c1 = I.Z.c;
                        %c1(map, :) = 0;
                        V1 = I.Z.V;
                        V1(map, :) = alpha*V1(map, :);
                        new_Z = Zono(c1, V1);
                    else
                        new_Z = [];
                    end
                    
                    In = Star(V, I.C, I.d, I.predicate_lb, I.predicate_ub, new_Z);                    
                    map = find(lb < 0 & ub > 0);
                    m = length(map);                    
                    for i=1:m
                        if strcmp(dis_opt, 'display')
                            fprintf('\nPerforming exact PosLin_%d operation using Star', map(i));
                        end
                        In = LReLU.stepReachMultipleInputs(In, map(i), option, lp_solver);
                    end               
                    S = In;
                end
                
            else
                S = [];
            end
        end
        
        function S = reach_star_exact_multipleInputs(varargin)
            switch nargin
                case 2
                    In = varargin{1};
                    option = varargin{2};
                    dis_opt = [];
                    lp_solver = 'linprog';
                case 3
                    In = varargin{1};
                    option = varargin{2};
                    dis_opt = varargin{3};
                    lp_solver = 'linprog';
                case 4
                    In = varargin{1};
                    option = varargin{2};
                    dis_opt = varargin{3};
                    lp_solver = varargin{4};
                otherwise
                    error('Invalid number of input arguments, should be 2, 3 or 4');
            end
            
             alpha = 0.01;
             n = length(In);
             S = [];
             if strcmp(option, 'parallel')
                 parfor i=1:n
                     S = [S LReLU.reach_star_exact(In(i), [], dis_opt, lp_solver)];
                 end
             elseif isempty(option) || strcmp(option, 'single')
                 for i=1:n
                     S = [S LReLU.reach_star_exact(In(i), [], dis_opt, lp_solver)];
                 end
             else
                 error('Unknown computation option');
             end
        end
        
        %step reach approximation using star
        function S = stepReachStarApprox(I, index)
            if ~isa(I, 'Star')
                error('Input is not a star');
            end
            alpha = 0.01;            
            lb = I.getMin(index);
              
            if lb > 0
                S = I;
            else
                ub = I.getMax(index);
                if ub <= 0
                    V = I.V;
                    V(index, :) = alpha*V(index, :);
                    if ~isempty(I.Z)
                        c1= I.Z.c;
                        %c1(index) = 0;
                        V1 = I.Z.V;
                        V1(index, :) = alpha*V1(index, :);
                        new_Z = Zono(c1, V1); % update outer-zono
                    else
                        new_Z = [];
                    end
                    S = Star(V, I.C, I.d, I.predicate_lb, I.predicate_ub, new_Z);
                else
                    fprintf('\nAdd a new predicate variables at index = %d', index);
                    n = I.nVar + 1;
                    % over-approximation constraints 
                    % constraint 1: y[index] >= alpha*x[index]
                    C1 = [alpha*I.V(index, 2:n) -1];
                    d1 = -I.V(index, 1);
                    % constraint 2: y[index] >= x[index]
                    C2 = [I.V(index, 2:n) -1];
                    d2 = -I.V(index, 1);
                    % constraint 3: y[index] <= alpha*lb + (ub - alpha*lb)*(x[index] -lb)/(ub - lb)
                    C3 = [-((ub-alpha*lb)/(ub-lb))*I.V(index, 2:n) 1];
                    d3 = alpha*lb -(ub-alpha*lb)*lb/(ub-lb) + (ub-alpha*lb)*I.V(index, 1)/(ub-lb);

                    m = size(I.C, 1);
                    C0 = [I.C zeros(m, 1)];
                    d0 = I.d;
                    new_C = [C0; C1; C2; C3];
                    new_d = [d0; d1; d2; d3];
                    new_V = [I.V zeros(I.dim, 1)];
                    new_V(index, :) = zeros(1, n+1);
                    new_V(index, n+1) = 1;     
                    
                    if isempty(I.predicate_lb) || isempty(I.predicate_ub)
                        [pred_lb, pred_ub] = I.getPredicateBounds;
                    else
                        pred_lb = I.predicate_lb;
                        pred_ub = I.predicate_ub;
                    end
                    new_predicate_lb = [pred_lb; 0];                
                    new_predicate_ub = [pred_ub; ub];
                    
                    % update outer-zono
                    lamda = (ub-alpha*lb)/(ub -lb);
                    mu = -0.5*(alpha*lb -(ub-alpha*lb)*lb/(ub-lb));
                    if ~isempty(I.Z)
                        c = I.Z.c; 
                        c(index) = lamda * c(index) + mu;
                        V = I.Z.V;
                        V(index, :) = lamda * V(index, :);
                        I1 = zeros(I.dim,1);
                        I1(index) = mu;
                        V = [V I1];              
                        new_Z = Zono(c, V);
                    else
                        new_Z = [];
                    end
                    
                    S = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub, new_Z);
                end
            end     
        end
        
        % over-approximate reachability analysis using Star
        function S = reach_star_approx(I)
            % @I: star input set
            % @S: star output set


            if ~isa(I, 'Star')
                error('Input is not a star');
            end
            alpha = 0.01;
            if isempty(I)
                S = [];
            else
                [lb, ub] = I.estimateRanges;
                if isempty(lb) || isempty(ub)
                    S = [];
                else
                    map = find(ub <= 0); % computation map
                    V = I.V;
                    V(map, :) = alpha*V(map, :);
                    % update outer-zono
                    if ~isempty(I.Z)
                        c1 = I.Z.c;
                        %c1(map, :) = 0;
                        V1 = I.Z.V;
                        V1(map, :) = alpha*V1(map, :);
                        new_Z = Zono(c1, V1);
                    else
                        new_Z = [];
                    end
                    
                    In = Star(V, I.C, I.d, I.predicate_lb, I.predicate_ub, new_Z);                    
                    map = find(lb < 0 & ub > 0);
                    m = length(map);                    
                    for i=1:m
                        fprintf('\nPerforming exact LeakyRelu_%d operation using Star', map(i));
                        In = LReLU.stepReachStarApprox(In, map(i));
                    end               
                    S = In;
                end
            end
        end
        
        % step reach approximation using star
        function S = multipleStepReachStarApprox_at_one(I, index, lb, ub)
            if ~isa(I, 'Star')
                error('Input is not a star');
            end
            alpha = 0.01;
            N = I.dim; 
            m = length(index); % number of neurons involved (number of new predicate variables introduced)
            
            % construct new basis array
            V1 = I.V; % originial basis array
            V1(index, :) = 0;
            V2 = zeros(N, m); % basis array for new predicates
            for i=1:m
                V2(index(i), i) = 1;
            end
            new_V = [V1 V2]; % new basis for over-approximate star set
            
            % construct new constraints on new predicate variables
            
            % case 0: keep the old constraints on the old predicate
            % variables
            n = I.nVar; % number of old predicate variables
            C0 = [I.C zeros(size(I.C, 1), m)];
            d0 = I.d; 
            
            % case 1: y[index] >= alpha*x[index]
            C1 = [alpha*I.V(index, 2:n+1) -V2(index, 1:m)];
            d1 = -I.V(index, 1);
            
            %case 2: y(index) >= x(index)
            C2 = [I.V(index, 2:n+1) -V2(index, 1:m)];
            d2 = -I.V(index, 1);
            
            % case 3: y[index] <= alpha*lb + (ub - alpha*lb)*(x[index] -lb)/(ub - lb)
            a = (ub-alpha*lb)./(ub-lb); % divide element-wise
            b = a.*lb; % multiply element-wise
            C3 = [-a.*I.V(index, 2:n+1) V2(index, 1:m)];
            d3 = alpha*lb -b + a.*I.V(index, 1);

            new_C = [C0; C1; C2; C3];
            new_d = [d0; d1; d2; d3];
            
            new_pred_lb = [I.predicate_lb; zeros(m,1)];
            new_pred_ub = [I.predicate_ub; ub];
            
            S = Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub);
        end
        
        % more efficient method by doing multiple stepReach at one time
        % over-approximate reachability analysis using Star
        function S = reach_star_approx2(varargin)
            
            switch nargin
                case 1
                    I = varargin{1};
                    option = 'single';
                    dis_opt = [];
                    lp_solver = 'linprog';
                case 2
                    I = varargin{1};
                    option = varargin{2};
                    dis_opt = [];
                    lp_solver = 'linprog';
                case 3
                    I = varargin{1};
                    option = varargin{2};
                    dis_opt = varargin{3};
                    lp_solver = 'linprog';
                case 4
                    I = varargin{1};
                    option = varargin{2};
                    dis_opt = varargin{3};
                    lp_solver = varargin{4};
                otherwise
                    error('Invalid number of input arguments, should be 1, 2, 3, or 4');
            end

            if ~isa(I, 'Star')
                error('Input is not a star');
            end
            alpha = 0.01;
            if isempty(I)
                S = [];
            else
                [lb, ub] = I.estimateRanges;
                if isempty(lb) || isempty(ub)
                    S = [];
                else
                    if strcmp(dis_opt, 'display')
                        fprintf('\nFinding all neurons (in %d neurons) with ub <= 0...', length(ub));
                    end
                    map1 = find(ub <= 0); % computation map
                    if strcmp(dis_opt, 'display')
                        fprintf('\n%d neurons with ub <= 0 are found by estimating ranges', length(map1));
                    end
                    
                    map2 = find(lb < 0 & ub > 0);
                    if strcmp(dis_opt, 'display')
                        fprintf('\nFinding neurons (in %d neurons) with ub <= 0 by optimizing ranges: ', length(map2));
                    end
                    xmax = I.getMaxs(map2, option, dis_opt, lp_solver);
                    map3 = find(xmax <= 0);
                    if strcmp(dis_opt, 'display')
                        fprintf('\n%d neurons (in %d neurons) with ub <= 0 are found by optimizing ranges', length(map3), length(map2));
                    end
                    n = length(map3);
                    map4 = zeros(n,1);
                    for i=1:n
                        map4(i) = map2(map3(i));
                    end
                    map11 = [map1; map4];
                    %replace map11 indexes with alpha*x
                    V = I.V;
                    V(map11, :) = alpha*V(map11, :);
                    % update outer-zono
                    if ~isempty(I.Z)
                        c1 = I.Z.c;
                        %c1(map, :) = 0;
                        V1 = I.Z.V;
                        V1(map11, :) = alpha*V1(map11, :);
                        new_Z = Zono(c1, V1);
                    else
                        new_Z = [];
                    end
                    
                    In = Star(V, I.C, I.d, I.predicate_lb, I.predicate_ub, new_Z);
                    if strcmp(dis_opt, 'display')
                        fprintf('\n(%d+%d =%d)/%d neurons have ub <= 0', length(map1), length(map3), length(map11), length(ub));
                    end
                    
                    % find all indexes that have lb < 0 & ub > 0, then
                    % apply the over-approximation rule for ReLU
                    if strcmp(dis_opt, 'display')
                        fprintf("\nFinding all neurons (in %d neurons) with lb < 0 & ub >0: ", length(ub));
                    end
                    map5 = find(xmax > 0);
                    map6 = map2(map5(:)); % all indexes having ub > 0
                    xmax1 = xmax(map5(:)); % upper bound of all neurons having ub > 0
                    xmin = I.getMins(map6, option, dis_opt, lp_solver); 
                    map7 = find(xmin < 0); 
                    map8 = map6(map7(:)); % all indexes having lb < 0 & ub > 0
                    lb1 = xmin(map7(:));  % lower bound of all indexes having lb < 0 & ub > 0
                    ub1 = xmax1(map7(:)); % upper bound of all neurons having lb < 0 & ub > 0
                    
                    if strcmp(dis_opt, 'display')
                        fprintf('\n%d/%d neurons have lb < 0 & ub > 0', length(map8), length(ub));
                        fprintf('\nConstruct new star set, %d new predicate variables are introduced', length(map8));
                    end
                    S = LReLU.multipleStepReachStarApprox_at_one(In, map8, lb1, ub1); % one-shot approximation
                end
            end
        end     
    end
    
    % reachability analysis using Polyhedron
    methods(Static) 
        function R = stepReach_Polyhedron(I, index, xmin, xmax)
            alpha = 0.01;
            I.normalize;
            dim = I.Dim;
            if xmin >= 0
                R = I; 
            elseif xmax < 0 
                Im = eye(dim);
                Im(index, index) = alpha*Im(index,index);
                R = Im*I;
            elseif xmin < 0 && xmax >= 0
                Z1 = zeros(1, dim);
                Z1(1, index) = 1;
                Z2 = zeros(1, dim);
                Z2(1, index) = -1;

                A1 = vertcat(I.A, Z1);
                A2 = vertcat(I.A, Z2);
                b  = vertcat(I.b, [0]);
                R1 = Polyhedron('A', A1, 'b', b, 'Ae', I.Ae, 'be', I.be);
                R2 = Polyhedron('A', A2, 'b', b, 'Ae', I.Ae, 'be', I.be);
                
                Im = eye(dim);
                Im(index, index) = alpha*Im(index,index);
                R1 = Im*R1
                if R1.isEmptySet 
                    if R2.isEmptySet
                        R = [];
                    else
                        
                        R = R2;
                    end
                else
                    if R2.isEmptySet
                        R = R1;
                    else
                        R = [R1 R2];
                    end
                    
                end 
            end
        end
    
        % stepReach for multiple Input Sets 
        function R = stepReachMultipleInputs_Polyhedron(varargin)
            switch nargin
                
                case 5
                    I = varargin{1};
                    index = varargin{2};
                    xmin = varargin{3};
                    xmax = varargin{4};
                    option = varargin{5};
                
                otherwise
                    error('Invalid number of input arguments (should be 5)');
            end
            p = length(I);
            R = [];
            if isempty(option)
                for i=1:p
                    R =[R, LReLU.stepReach_Polyhedron(I(i), index, xmin, xmax)];
                end
            elseif strcmp(option, 'parallel')
                parfor i=1:p
                    R =[R, LReLU.stepReach_Polyhedron(I(i), index, xmin, xmax)];
                end
                
            else
                error('Unknown option');
            end            
        end
    
        % exact reachability analysis using Polyhedron
        function R = reach_polyhedron_exact(varargin)
            switch nargin
                case 2
                    I = varargin{1};
                    option = varargin{2};
                    dis_opt = [];
                case 3
                    I = varargin{1};
                    option = varargin{2};
                    dis_opt = varargin{3};
                otherwise
                    error('Invalid number of input arguments, should be 2 or 3');
            end
            if ~isempty(I)
                if isa(I, 'Polyhedron')            
                    I.outerApprox; % find bounds of I state vector
                    lb = I.Internal.lb; % min-vec of x vector
                    ub = I.Internal.ub; % max-vec of x vector
                else
                    error('Input set is not a Polyhedron');
                end
                
                if isempty(lb) || isempty(ub)
                    R = [];
                else
                    map = find(lb < 0); % computation map
                    m = size(map, 1); % number of stepReach operations needs to be executed
                    In = I;
                    for i=1:m
                        if strcmp(dis_opt, 'display')
                            fprintf('\nPerforming exact LeakyReLU%d operation using Polyhedron', map(i));
                        end
                        In = LReLU.stepReachMultipleInputs_Polyhedron(In, map(i), lb(map(i)), ub(map(i)), option);
                    end               
                    R = In;
                end
                
            else
                R = [];
            end
        end
    end
    
    %over-approximate reachability analysis use zonotope
    methods(Static) 
        
        % step over-approximate reachability analysis using zonotope
        function Z = stepReachZonoApprox(I, index, lb, ub)
            if ~isa(I, 'Zono')
                error('Input is not a Zonotope');
            end
            alpha = 0.01;
            if lb >= 0
                Z = Zono(I.c, I.V);
            elseif ub <= 0
                c = I.c;
                %c(index) = 0;
                V = I.V;
                V(index, :) = alpha*V(index, :);
                Z = Zono(c, V);
            elseif lb < 0 && ub > 0
                lamda = (ub-alpha*lb)/(ub -lb);
                mu = -0.5*(1-alpha)*ub*lb/(ub-lb);               
                
                c = I.c; 
                c(index) = lamda * c(index) + mu;
                V = I.V;
                V(index, :) = lamda * V(index, :);
                I1 = zeros(I.dim,1);
                I1(index) = mu;
                V = [V I1];
                
                Z = Zono(c, V);                
            end
        end
        % over-approximate reachability analysis use zonotope
        function Z = reach_zono_approx(varargin)
            % @I: zonotope input
            % @Z: zonotope output
            
            switch nargin
                case 1
                    I = varargin{1};
                    dis_opt = [];
                case 2
                    I = varargin{1};
                    dis_opt = varargin{2};
                otherwise
                    error('Invalid number of input arguments, should be 1 or 2');
            end
            
            if ~isa(I, 'Zono')
                error('Input is not a Zonotope');
            end
                      
            In = I;
            [lb, ub] = I.getBounds;
            for i=1:I.dim
                if strcmp(dis_opt, 'display')
                    fprintf('\nPerforming approximate LeakyRelu_%d operation using Zonotope', i);
                end
                In = LReLU.stepReachZonoApprox(In, i, lb(i), ub(i));
            end
            Z = In;
                       
        end
    end
    
    % over-approximate reachability analysis using abstract-domain
    methods(Static)
        
        % step over-approximate reachability analysis using abstract-domain
        % we use star set to represent abstract-domain
        function S = stepReachAbstractDomain(varargin)
            switch nargin
                
                case 4
                    
                    I = varargin{1};
                    index = varargin{2};
                    lb = varargin{3};
                    ub = varargin{4};
                
                case 2
                    I = varargin{1};
                    index = varargin{2};
                    [lb, ub] = I.getRange(index); % our improved approach
                    %[lb, ub] = I.estimateRange(index); % originial DeepPoly approach use estimated range
                otherwise
                    error('Invalid number of input arguments (should be 2 or 4)');
            end
            if ~isa(I, 'Star')
                error('Input is not a Star');
            end
            alpha = 0.01;
            if lb > 0
                S = Star(I.V, I.C, I.d, I.predicate_lb, I.predicate_ub);
            elseif ub <= 0
                V = I.V;
                V(index, :) = alpha*V(index, :);
                S = Star(V, I.C, I.d, I.predicate_lb, I.predicate_ub);
            elseif lb <=0 && ub >= 0
                
                S1 = (1-alpha)*ub*(ub-lb)/2; % area of the first candidate abstract-domain
                S2 = -(1-alpha)*lb*(ub-lb)/2; % area of the second candidate abstract-domain  
                
                n = I.nVar + 1;
                
                % constraint 1: y[index] >= alpha*x[index]
                C1 = [alpha*I.V(index, 2:n) -1];
                d1 = -I.V(index, 1);
                % constraint 2: y[index] >= x[index]
                C2 = [I.V(index, 2:n) -1];
                d2 = -I.V(index, 1);
                % constraint 3: y[index] <= alpha*lb + (ub - alpha*lb)*(x[index] -lb)/(ub - lb)
                C3 = [-((ub-alpha*lb)/(ub-lb))*I.V(index, 2:n) 1];
                d3 = alpha*lb -(ub-alpha*lb)*lb/(ub-lb) + (ub-alpha*lb)*I.V(index, 1)/(ub-lb);
                
                m = size(I.C, 1);
                C0 = [I.C zeros(m, 1)];
                d0 = I.d;
                new_V = [I.V zeros(I.dim, 1)];
                new_V(index, :) = zeros(1, n+1);
                new_V(index, n+1) = 1;
                
                if isempty(I.predicate_lb) || isempty(I.predicate_ub)
                    [pred_lb, pred_ub] = I.getPredicateBounds;
                else
                    pred_lb = I.predicate_lb;
                    pred_ub = I.predicate_ub;
                end
                
                if S1 < S2 %select constraints corresponding to S1
                    %get first cadidate as resulted abstract-domain
                    new_C = [C0; C1; C3];
                    new_d = [d0; d1; d3];
                    new_pred_lb = [pred_lb; alpha*lb];
                    new_pred_ub = [pred_ub; ub];
                else
                    % choose the second candidate as the abstract-domain                                      
                    new_C = [C0; C2; C3];
                    new_d = [d0; d2; d3];
                    new_pred_lb = [pred_lb; lb];
                    new_pred_ub = [pred_ub; ub];
                end
                
                 S = Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub);
            end
        end
    
        % over-approximate reachability analysis using abstract-domain
        function S = reach_abstract_domain(varargin)
             switch nargin
                case 1
                    I = varargin{1};
                    dis_opt = [];
                case 2
                    I = varargin{1};
                    dis_opt = varargin{2};
                otherwise
                    error('Invalid number of input arguments, should be 1 or 2');
            end
            
            alpha = 0.01;
            if ~isa(I, 'Star')
                error('Input is not a star');
            end

            if isempty(I)
                S = [];
            else    
                %[lb, ub] = I.estimateRanges;
                [lb, ub] = I.getRanges(); % get tightest ranges from LP optimization
                if isempty(lb) || isempty(ub)
                    S = [];
                else
                    map = find(ub <= 0); % computation map
                    V = I.V;
                    V(map, :) = alpha*V(map, :);
                    % update outer-zono
                    if ~isempty(I.Z)
                        c1 = I.Z.c;
                        %c1(map, :) = 0;
                        V1 = I.Z.V;
                        V1(map, :) = alpha*V1(map, :);
                        new_Z = Zono(c1, V1);
                    else
                        new_Z = [];
                    end
                    
                    In = Star(V, I.C, I.d, I.predicate_lb, I.predicate_ub, new_Z);                    
                    map = find(lb < 0 & ub > 0);
                    m = length(map);                    
                    for i=1:m
                        if strcmp(dis_opt, 'display')
                            fprintf('\nPerforming exact PosLin_%d operation using Abstract Domain', map(i));
                        end
                        In = LReLU.stepReachAbstractDomain(In, map(i), lb(map(i)), ub(map(i)));
                    end               
                    S = In;
                end
            end
        end  
    end
    
    methods(Static) %main reach method
        %main function for reachability analysis
        function R = reach(varargin)
            % @I: an array of star input sets
            % @method: 'exact-star' or 'approx-star' or 'approx-zono' or
            % 'abs-dom' or 'face-latice'
            % @option: = 'parallel' use parallel option
            %          = '' do use parallel option
            
            % author: Dung Tran
            % date: 27/2/2019
            % update: 7/15/2020: add display option
            
            switch nargin
                
                case 6
                    I = varargin{1};
                    method = varargin{2};
                    option = varargin{3};
                    relaxFactor = varargin{4}; % used for aprox-star only
                    dis_opt = varargin{5}; % display option
                    lp_solver = varargin{6}; 
                
                case 5
                    I = varargin{1};
                    method = varargin{2};
                    option = varargin{3};
                    relaxFactor = varargin{4}; % used for aprox-star only
                    dis_opt = varargin{5}; % display option
                    lp_solver = 'linprog';
                
                case 4
                    I = varargin{1};
                    method = varargin{2};
                    option = varargin{3};
                    relaxFactor = varargin{4}; % used for aprox-star only
                    dis_opt = [];
                    lp_solver = 'linprog';
                                    
                case 3
                    I = varargin{1};
                    method = varargin{2};
                    option = varargin{3};
                    relaxFactor = 0; % used for aprox-star only
                    dis_opt = [];
                    lp_solver = 'linprog';
                case 2
                    I = varargin{1};
                    method = varargin{2};
                    option = 'parallel';
                    relaxFactor = 0; % used for aprox-star only
                    dis_opt = [];
                    lp_solver = 'linprog';
                case 1
                    I = varargin{1};
                    method = 'exact-star';
                    option = 'parallel';
                    relaxFactor = 0; % used for aprox-star only
                    dis_opt = [];
                    lp_solver = 'linprog';
                otherwise
                    error('Invalid number of input arguments (should be 1, 2, 3, 4, or 5)');
            end
            
            
            if strcmp(method, 'exact-star') % exact analysis using star
                R = LReLU.reach_star_exact_multipleInputs(I, option, dis_opt, lp_solver);
            elseif strcmp(method, 'exact-polyhedron') % exact analysis using polyhedron
                R = LReLU.reach_polyhedron_exact(I, option, dis_opt);
            elseif strcmp(method, 'approx-star')  % over-approximate analysis using star
                R = LReLU.reach_star_approx2(I, option, dis_opt, lp_solver);
%             elseif strcmp(method, 'relax-star-dis')
%                 R = LReLU.reach_relaxed_star_dis(I, relaxFactor, option, dis_opt, lp_solver);
%             elseif strcmp(method, 'relax-star-lb-ub')
%                 R = LReLU.reach_relaxed_star_lb_ub(I, relaxFactor, option, dis_opt, lp_solver);
%             elseif strcmp(method, 'relax-star-area')
%                 R = LReLU.reach_relaxed_star_area(I, relaxFactor, option, dis_opt, lp_solver);
%             elseif strcmp(method, 'relax-star-ub')
%                 R = LReLU.reach_relaxed_star_ub(I, relaxFactor, option, dis_opt, lp_solver);
%             elseif strcmp(method, 'relax-star-random')
%                 R = LReLU.reach_relaxed_star_random(I, relaxFactor, option, dis_opt, lp_solver);
%             elseif strcmp(method, 'relax-star-static')
%                 R = LReLU.reach_relaxed_star_static(I, relaxFactor, option, dis_opt, lp_solver);
            elseif strcmp(method, 'approx-zono')  % over-approximate analysis using zonotope 
                R = LReLU.reach_zono_approx(I, dis_opt);
            elseif strcmp(method, 'abs-dom')  % over-approximate analysis using abstract-domain
                R = LReLU.reach_abstract_domain(I, dis_opt);
            elseif strcmp(method, 'exact-face-latice') % exact analysis using face-latice
                fprintf('\nNNV does not yet support Exact Face-Latice Method');
            elseif strcmp(method, 'approx-face-latice') % over-approximate analysis using face-latice
                fprintf('\nNNV does not yet support Approximate Face-Latice Method');
            end
                            
        end
    end
 end