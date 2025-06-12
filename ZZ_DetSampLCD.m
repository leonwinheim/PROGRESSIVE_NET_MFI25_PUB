%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file is part of the following publication:
% Daniel Frisch (2021) Deterministic Sampling LCD [Source Code]. https://doi.org/10.24433/CO.9883102.v1
% Find it on CodeOcean: https://codeocean.com/capsule/1886845/tree/v1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Deterministic Sampling with the LCD 
% 
% Draw deterministic Gaussian samples based on the LCD. 
% Use the Nonlinear Estimation Toolbox developed at ISAS. 
%

% Dependencies 
%[flist,plist] = matlab.codetools.requiredFilesAndProducts('DetSampLCD.m'); [flist'; {plist.Name}']
%nonlinearestimation Toolbox, 

% Nonlinear Estimation Toolbox
%toolbox_path = '../toolbox/Matlab/Toolbox';
%assert(exist(toolbox_path,'dir')==7)
%addpath(genpath(toolbox_path))

% Sampling parameters
%L = 15; % number of samples
%dim = 2; % dimensionality 

L = 7500; % number of samples
dim = 61; % dimensionality 

% Wanted Gaussian parameters
%mu = [0;0]; % [dim x 1] mean vector
%C  = [2.^2, .7; .7, .6^2]; % [dim x dim] covariance matrix

mu = zeros(dim, 1);      % [dim x 1] zero mean vector
C  = eye(dim);           % [dim x dim] identity matrix (standard normal)

% Print to console
name = sprintf('DetSamplingLCD_dim=%u_L=%u', dim, L); 
fprintf('Parameters %s: \nL=%u   (number of samples) \ndim=%u   (dimension)  \nmu=%s   (mean)  \nC=%s   (covariance) \n\n', name, L, dim, mat2str(mu), mat2str(C))

% Draw Deterministic Samples, Standard normally distributed (mean=0, cov=I)
fprintf('Draw deterministic samples via LCD... \n')
lcd = GaussianSamplingLCD(); 
lcd.setNumSamples(L); 
samples_stdnormal = lcd.getStdNormalSamples(dim); % [dim x L] sample coordinates

% Transformation to Arbitrary Gaussian (mean=mu, cov=C) 
assert(isequal(size(mu),[dim,1]))
assert(isequal(size(C),[dim,dim]))
samples_gauss = chol(C)' * samples_stdnormal + mu; % [dim x L] sample coordinates

% Print to Console
%fprintf('\nDeterministic Gaussian Samples, standard normally distributed [dim x L]: \n%s\n', mat2str(samples_stdnormal,5))
%fprintf('\nDeterministic Gaussian Samples, distributed according to mu and C [dim x L]: \n%s\n', mat2str(samples_gauss,5))

% Export .csv
filename = fullfile([name,'.csv']);
fprintf('\nWrite std normal samples to csv:  %s. \n', filename)
writematrix(samples_stdnormal', filename)


% Plot (only for dim=2)
if dim~=2
  return
end
%fig = figure(); % use new figure every time 
fig = figure(582343); % use same figure when evaluated repeatedly (change number randomly) 
clf(fig); % if same fig window is reused: clear it  
set(fig, 'NumberTitle','off', 'Name',name, 'Color','white') % title bar; transparent  
ax = axes(fig); % create axes handle  
set(ax, 'Color','none', 'TickLabelInterpreter','LaTeX') % transparent ax; latex 
ax.NextPlot = 'add'; % hold on 
set([ax.XLabel,ax.YLabel,ax.ZLabel], 'Interpreter','LaTeX') % latex 
set(ax, 'XGrid','on', 'YGrid','on', 'ZGrid','on', 'XMinorGrid','on', 'YMinorGrid','on', 'ZMinorGrid','on')  
set(ax, 'MinorGridLineStyle','-', 'GridColor','black', 'MinorGridColor','black') 
set(ax, 'GridAlpha',.15, 'MinorGridAlpha',.05) 
set([ax.XAxis,ax.YAxis,ax.ZAxis], 'Color','black') % axes black instead of grey: looks sharp when printing 
axis(ax, 'equal') % keep equal proportions along all axes
ax.XAxis.Label.String = 'x';
ax.YAxis.Label.String = 'y';

% Extent to plot
extent = [min(samples_gauss,[],2), max(samples_gauss,[],2)]; 
fac = .3;
extent = [extent(:,1)-diff(extent,[],2)*fac, extent(:,2)+diff(extent,[],2)*fac];
ax.XAxis.Limits = extent(1,:);
ax.YAxis.Limits = extent(2,:);

% Samples
hs = scatter(ax, samples_gauss(1,:), samples_gauss(2,:), 'DisplayName','Deterministic Samples');
set(hs, 'Marker','.', 'SizeData',300)

% Gaussian Ellipse
fun = @(x,y) reshape( dot(([x(:)';y(:)']-mu),(C\([x(:)';y(:)']-mu)),1), size(x));
hg = fcontour(ax, fun, 'LevelList',[1 2 3 4 5], 'LineColor',[1 1 1]*.6, 'DisplayName','Gaussian Density $1\sigma, 2\sigma ...$');
set(hg, 'XRange',extent(1,:), 'YRange',extent(2,:))

% Legend
lg = legend(ax, [hs,hg]);
lg.Location = 'NorthOutside';
lg.Interpreter = 'LaTeX';

% Save figure
% Use export_fig to remove whitespace
filename = fullfile(name);
fprintf('Save plot as png:  %s. \n', filename)
print(fig, filename,'-dpng')

