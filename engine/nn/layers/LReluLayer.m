classdef LReluLayer < handle
    properties
        Name = 'leakyrelu_layer';
        alpha = 0.01;
        NumInputs = 1;
        InputNames = {'in'};
        NumOutputs = 1;
        OutputNames = {'out'};
    end
    
    % setting hyperparameters method
    methods
        % constructor of the class
        function obj = LReluLayer(varargin)  
            switch nargin
                case 6 % used for parsin a matlab relu layer
                    obj.Name = varargin{1};
                    obj.NumInputs = varargin{2};
                    obj.alpha = varargin{3};
                    obj.InputNames = varargin{4};
                    obj.NumOutputs = varargin{5};
                    obj.OutputNames = varargin{6};

                case 1
                    name = varargin{1};
                    if ~ischar(name)
                        error('Name is not char');
                    else
                        obj.Name = name;
                    end                    
                case 0
                    obj.Name = 'leakyrelu_layer';
                    obj.alpha = 0.01;
                otherwise
                    error('Invalid number of inputs (should be 0 or 1)');
            end 
        end
    end
    
    % evaluation method
    methods
        function y = evaluate(obj, input)
            % @input: 2 or 3-dimensional array, for example, input(:, :, :), 
            % @y: 2 or 3-dimensional array, for example, y(:, :, :)
            % @y: high-dimensional array (output volume)
            
            n = size(input);
            N = 1;
            for i=1:length(n)
                N = N*n(i);
            end
            
            I = reshape(input, [N 1]);
            y = LReLU.evaluate(I, obj.alpha);
            y = reshape(y, n);
        end
    end
    
    methods % reachability methods
       
        % reachability using ImageStar
        function images = reach_star_single_input(~, in_image, method, relaxFactor)
            % @in_image: an ImageStar input set
            % @method: = 'exact-star' or 'approx-star' or 'abs-dom'
            % @relaxFactor: of approx-star method
            % @images: an array of ImageStar (if we use 'exact-star' method)
            %         or a single ImageStar set
            
            if ~isa(in_image, 'ImageStar')
                error('input is not an ImageStar');
            end
            
            
            h = in_image.height;
            w = in_image.width;
            c = in_image.numChannel;
            
            Y = LReLU.reach(in_image.toStar, method, [], relaxFactor); % reachable set computation with ReLU
            n = length(Y);
            images(n) = ImageStar;
            % transform back to ImageStar
            for i=1:n
                images(i) = Y(i).toImageStar(h,w,c);
            end

        end
        
        % hangling multiple inputs
        function images = reach_star_multipleInputs(obj, in_images, method, option, relaxFactor)
            % @in_images: an array of ImageStars
            % @method: = 'exact-star' or 'approx-star' or 'abs-dom'
            % @option: = 'parallel' or 'single' or empty
            % @relaxFactor: of approx-star method
            % @images: an array of ImageStar (if we use 'exact-star' method)
            %         or a single ImageStar set
            
            images = [];
            n = length(in_images);
                        
            if strcmp(option, 'parallel')
                parfor i=1:n
                    images = [images obj.reach_star_single_input(in_images(i), method, relaxFactor)];
                end
            elseif strcmp(option, 'single') || isempty(option)
                for i=1:n
                    images = [images obj.reach_star_single_input(in_images(i), method, relaxFactor)];
                end
            else
                error('Unknown computation option');

            end
            
        end
        
        % reachability using ImageStar
        function images = reach_star_single_input2(~, in_image, method, option, relaxFactor, dis_opt, lp_solver)
            % @in_image: an ImageStar input set
            % @method: = 'exact-star' or 'approx-star' or 'abs-dom'
            % @relaxFactor: of approx-star method
            % @dis_opt: display option = [] or 'display'
            % @lp_solver: lp solver
            % @images: an array of ImageStar (if we use 'exact-star' method)
            %         or a single ImageStar set
                        
            if ~isa(in_image, 'ImageStar')
                error('input is not an ImageStar');
            end
            
            
            h = in_image.height;
            w = in_image.width;
            c = in_image.numChannel;
            
            Y = LReLU.reach(in_image.toStar, method, option, relaxFactor, dis_opt, lp_solver); % reachable set computation with ReLU
            n = length(Y);
            images(n) = ImageStar;
            % transform back to ImageStar
            for i=1:n
                images(i) = Y(i).toImageStar(h,w,c);
            end

        end
        
        % hangling multiple inputs
        function images = reach_star_multipleInputs2(obj, in_images, method, option, relaxFactor, dis_opt, lp_solver)
            % @in_images: an array of ImageStars
            % @method: = 'exact-star' or 'approx-star' or 'abs-dom'
            % @option: = 'parallel' or 'single' or empty
            % @relaxFactor: of approx-star method
            % @dis_opt: display option = [] or 'display'
            % @lp_solver: lp solver
            % @images: an array of ImageStar (if we use 'exact-star' method)
            %         or a single ImageStar set
            
            images = [];
            n = length(in_images);
                        
            for i=1:n
                images = [images obj.reach_star_single_input2(in_images(i), method, option, relaxFactor, dis_opt, lp_solver)];
            end
            
        end
        
        % reachability using ImageZono
        function image = reach_zono(~, in_image)
            % @in_image: an ImageZono input set
            
            if ~isa(in_image, 'ImageZono')
                error('input is not an ImageZono');
            end
            
            h = in_image.height;
            w = in_image.width;
            c = in_image.numChannels;
            In = in_image.toZono;
            Y = LReLU.reach(In, 'approx-zono');
            image = Y.toImageZono(h,w,c);
            
        end
        
        % handling multiple inputs
        function images = reach_zono_multipleInputs(obj, in_images, option)
            % @in_images: an array of ImageStars
            % @option: = 'parallel' or 'single' or empty
            % @images: an array of ImageStar (if we use 'exact-star' method)
            %         or a single ImageStar set
            
            n = length(in_images);
            images(n) = ImageZono;
                        
            if strcmp(option, 'parallel')
                parfor i=1:n
                    images(i) = obj.reach_zono(in_images(i));
                end
            elseif strcmp(option, 'single') || isempty(option)
                for i=1:n
                    images(i) = obj.reach_zono(in_images(i));
                end
            else
                error('Unknown computation option');
            end
            
        end
        
        % MAIN REACHABILITY METHOD
        function images = reach(varargin)
            % @in_image: an input imagestar
            % @image: output set
            % @option: = 'single' or 'parallel' 
            switch nargin
                case 6
                    obj = varargin{1};
                    in_images = varargin{2};
                    method = varargin{3};
                    option = varargin{4};
                    dis_opt = varargin{5}; 
                    lp_solver = varargin{6}; 
                case 5
                    obj = varargin{1};
                    in_images = varargin{2};
                    method = varargin{3};
                    option = varargin{4};
                    dis_opt = varargin{5}; % display option
                    lp_solver = 'linprog';

    %                 case 5
    %                     obj = varargin{1};
    %                     in_images = varargin{2};
    %                     method = varargin{3};
    %                     option = varargin{4};
    %                     dis_opt = [];
    %                     lp_solver = 'linprog';

                case 4
                    obj = varargin{1};
                    in_images = varargin{2};
                    method = varargin{3};
                    option = varargin{4};
                    dis_opt = [];
                    lp_solver = 'linprog';
                case 3
                    obj = varargin{1};
                    in_images = varargin{2};
                    method = varargin{3};
                    option = 'single';
                    dis_opt = [];
                    lp_solver = 'linprog';
                otherwise
                    error('Invalid number of input arguments (should be 1, 2, 3, 4, 5, or 6)');
            end
            relaxFactor = 0;
            if strcmp(method, 'approx-star') || strcmp(method, 'exact-star') || strcmp(method, 'abs-dom')
                images = obj.reach_star_multipleInputs2(in_images, method, option, relaxFactor, dis_opt, lp_solver);%reach_star_multipleInputs2
            elseif strcmp(method, 'approx-zono')
                images = obj.reach_zono_multipleInputs(in_images, option);
            else
                error("Uknown reachability method");
            end
        end
    
    end
        
    
    methods(Static)
         % parse a trained relu Layer from matlab
        function L = parse(leakyRelu_layer)
            % @leakyrelu_layer: 
             
            if ~isa(leakyRelu_layer, 'nnet.cnn.layer.LReLULayer')
                error('Input is not a Matlab nnet.cnn.layer.leakyReLULayer class');
            end
            
            L = LReluLayer(leakyRelu_layer.Name, leakyRelu_layer.alpha, leakyRelu_layer.NumInputs, leakyRelu_layer.InputNames, leakyRelu_layer.NumOutputs, leakyRelu_layer.OutputNames);
            fprintf('\nParsing a Matlab relu layer is done successfully');
            
        end
        
    end
end