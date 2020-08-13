%% calculating edge weights

clear all;
close all;

% params
a = 8.5/2;
b = 6.5/2;
num_channels = 8;
theta = 2.5;
tau = 5.5;

chan1.x = a*cos(5*pi/4);
chan1.y = b*sin(5*pi/4);
chan2.x = a*cos(pi);
chan2.y = b*sin(pi);
chan3.x = a*cos(3*pi/4);
chan3.y = b*sin(3*pi/4);
chan4.x = a*cos(pi/2);
chan4.y = b*sin(pi/2);
chan5.x = a*cos(pi/4);
chan5.y = b*sin(pi/4);
chan6.x = a*cos(0);
chan6.y = b*sin(0);
chan7.x = a*cos(-pi/4);
chan7.y = b*sin(-pi/4);
chan8.x = a*cos(-pi/2);
chan8.y = b*sin(-pi/2);

nodes = [chan1, chan2, chan3, chan4, chan5, chan6, chan7, chan8];

% display the electrode configuration 
figure;
plot([nodes.x, nodes(1).x], [nodes.y, nodes(1).y], 'ro-');
title('electrode_config');

D = zeros(num_channels, num_channels); % initializing distance matrix

for i = 1:num_channels
    for j = 1:num_channels
        D(i, j) = sqrt((nodes(i).x - nodes(j).x)^2 + (nodes(i).y - nodes(j).y)^2);
    end
end

temp = exp(-(D.^2)/(2*theta^2));
temp(D > tau) = 0;
A = temp;

disp('Distnace Matrix : ');
disp(D);
disp('Adjacency Matrix : ');
disp(A);

load alt_adj_matrix.mat

G = graph(A, 'OmitSelfLoops');
G.Edges

% plot
figure; 
p = plot(G, 'Layout', 'circle', 'EdgeLabel',G.Edges.Weight); title('Graph');
p.NodeColor = 'r';
p.Marker = 's';
p.MarkerSize = 8;
p.LineWidth = 2;
colormap([[0 0 0];jet(256)]);          % select color palette
caxis([0 0.5]);
p.EdgeCData=G.Edges.Weight;    % define edge color
p.XData=[nodes.x];      % place node locations on plot
p.YData=[nodes.y];
colorbar



