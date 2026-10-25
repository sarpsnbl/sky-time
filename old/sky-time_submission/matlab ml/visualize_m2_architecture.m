%% visualize_m2_architecture.m
%  Draws the Method 2 (SVR + GBT + RF Ensemble) pipeline as a block diagram.
%  Output: m2_architecture.png  (saved in the current directory)

clear; close all;

fig = figure('Color','white','Position',[100 100 1100 420]);

% CRITICAL FIX: Set Position to [0 0 1 1] so the axes fill the figure. 
% This perfectly aligns the figure coordinates (0 to 1) with the axes math (x/11, y/4.2)
ax  = axes('Parent',fig,'Position',[0 0 1 1],'Visible','off');
axis(ax,[0 11 0 4.2]); hold on;

% ── Colour palette ────────────────────────────────────────────────────────
cInput  = [0.22 0.49 0.72];   % steel blue
cFeat   = [0.12 0.63 0.47];   % teal
cModel  = [0.83 0.37 0.15];   % burnt orange
cEns    = [0.50 0.18 0.56];   % purple
cOut    = [0.18 0.18 0.18];   % near-black

% ─────────────────────────────────────────────────────────────────────────
%  BLOCKS
% ─────────────────────────────────────────────────────────────────────────

% Input image
drawBox(0.9, 2.1, 1.3, 0.7, cInput, cInput*0.7, sprintf('Sky Image\n224×224'), 9);

% Feature extraction
drawBox(2.8, 2.7, 1.5, 0.65, cFeat, cFeat*0.7, sprintf('Spatial Grid\n54-dim'), 9);
drawBox(2.8, 2.1, 1.5, 0.65, cFeat, cFeat*0.7, sprintf('Global Photometric\n125-dim'), 9);
drawBox(2.8, 1.5, 1.5, 0.65, cFeat, cFeat*0.7, sprintf('Sun Proxy\n3-dim'), 9);

% Date encoding
drawBox(0.9, 0.75, 1.3, 0.65, cInput, cInput*0.7, sprintf('Calendar\nMetadata'), 9);
drawBox(2.8, 0.75, 1.5, 0.65, cFeat, cFeat*0.7, sprintf('Cyclic Encoding\n4-dim'), 9);

% Concatenation node
drawBox(4.7, 2.1, 0.9, 0.55, [0.55 0.55 0.55],[0.3 0.3 0.3], sprintf('Concat\n186-dim'), 8.5);

% Z-score
drawBox(5.95, 2.1, 0.9, 0.55, [0.45 0.45 0.55],[0.25 0.25 0.4], sprintf('Z-score\nnorm.'), 8.5);

% Three models — sin branch labels added below
yModels = [3.2, 2.1, 1.0];
modelNames = {'SVR (RBF)', 'GBT  500×', 'RF   500×'};
for k = 1:3
    drawBox(7.4, yModels(k), 1.25, 0.55, cModel, cModel*0.7, modelNames{k}, 9);
    % sin/cos indicator
    text(8.1, yModels(k)+0.04, '\rightarrow', 'FontSize',8,'Color',[0.7 0.7 0.7], ...
         'HorizontalAlignment','left');
end

% Ensemble average
drawBox(9.0, 2.1, 1.1, 0.55, cEns, cEns*0.7, sprintf('Ensemble\nAvg (sin/cos)'), 8.5);

% Output
drawBox(10.4, 2.1, 0.9, 0.55, cOut, [0 0 0], sprintf('atan2\nDecode'), 8.5);

% ─────────────────────────────────────────────────────────────────────────
%  BRACKET (Drawn with lines for OS consistency)
% ─────────────────────────────────────────────────────────────────────────
plot([1.9 1.75 1.75 1.9], [2.7 2.7 1.5 1.5], 'Color', [0.55 0.55 0.55], 'LineWidth', 1.5);
plot([1.75 1.6], [2.1 2.1], 'Color', [0.55 0.55 0.55], 'LineWidth', 1.5);

% ─────────────────────────────────────────────────────────────────────────
%  ARROWS
% ─────────────────────────────────────────────────────────────────────────

% Image → Bracket
drawArrow(1.55, 2.10, 1.60, 2.10);

% Bracket → three feature blocks
drawArrow(1.90, 2.70, 2.05, 2.70);
drawArrow(1.90, 2.10, 2.05, 2.10);
drawArrow(1.90, 1.50, 2.05, 1.50);

% Calendar → cyclic encoding
drawArrow(1.55, 0.75, 2.05, 0.75);

% Feature blocks + Cyclic encoding → concat
for yy = [2.7, 2.1, 1.5, 0.75]
    drawArrow(3.55, yy, 4.25, 2.1);
end

% Concat → Z-score
drawArrow(5.15, 2.1, 5.50, 2.1);

% Z-score → three models
for k = 1:3
    drawArrow(6.40, 2.1, 6.77, yModels(k));
end

% Models → ensemble
for k = 1:3
    drawArrow(8.02, yModels(k), 8.45, 2.1);
end

% Ensemble → decode
drawArrow(9.55, 2.1, 9.95, 2.1);

% ─────────────────────────────────────────────────────────────────────────
%  EDGE LABELS
% ─────────────────────────────────────────────────────────────────────────
edgeLabel(4.55, 1.45, '186-d');
edgeLabel(5.32, 2.25, 'z-scored');
edgeLabel(6.60, 2.80, 'sin / cos');
edgeLabel(8.25, 2.80, 'sin / cos');
edgeLabel(9.75, 2.25, 'time');

% ─────────────────────────────────────────────────────────────────────────
%  TITLE
% ─────────────────────────────────────────────────────────────────────────
% Replaced title() with text() since axes cover the whole screen
text(5.5, 3.9, 'Method 2 – Statistical Ensemble Pipeline', ...
      'FontName','Helvetica','FontSize',12,'FontWeight','bold',...
      'Color',[0.2 0.2 0.2],'HorizontalAlignment','center');

hold off;
axis off;

% ── Export ────────────────────────────────────────────────────────────────
exportgraphics(fig, 'm2_architecture.png', 'Resolution', 200);
fprintf('Saved:  m2_architecture.png\n');


% =========================================================================
%  LOCAL HELPER FUNCTIONS (Placed at the bottom for modern MATLAB compatibility)
% =========================================================================

% ── Helper: filled rounded rectangle ─────────────────────────────────────
function drawBox(x, y, w, h, fc, lc, txt, fsz)
    rectangle('Position',[x-w/2, y-h/2, w, h], ...
              'Curvature',[0.15 0.35], ...
              'FaceColor', fc, 'EdgeColor', lc, 'LineWidth', 1.6);
    % Using explicit [1 1 1] RGB triplet instead of the 'white' variable
    text(x, y, txt, 'HorizontalAlignment','center', ...
         'VerticalAlignment','middle', 'FontSize', fsz, ...
         'FontName','Helvetica','Color', [1 1 1], 'FontWeight','bold', ...
         'Interpreter','none');
end

% ── Helper: arrow ─────────────────────────────────────────────────────────
function drawArrow(x1,y1,x2,y2)
    annotation('arrow', ...
        [x1/11, x2/11], [y1/4.2, y2/4.2], ...
        'HeadWidth',8,'HeadLength',7,'Color',[0.35 0.35 0.35],'LineWidth',1.4);
end

% ── Helper: small label ───────────────────────────────────────────────────
function edgeLabel(x,y,txt)
    text(x,y,txt,'HorizontalAlignment','center','VerticalAlignment','middle', ...
         'FontSize',8,'FontName','Helvetica','Color',[0.4 0.4 0.4], ...
         'Interpreter','none');
end