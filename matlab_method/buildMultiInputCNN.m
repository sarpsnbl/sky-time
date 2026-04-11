function lgraph = buildMultiInputCNN(cfg)
% BUILDMULTIINPUTCNN  Build a multi-input regression network.
%
%   Architecture
%   ┌─────────────────────────────────────────────────────────┐
%   │  IMAGE INPUT [224×224×3]                                │
%   │   └─ SqueezeNet (frozen conv layers, trainable fire7-9) │
%   │       └─ pool10  → flatten  [512]                       │
%   │                         ╲                               │
%   │  DATE INPUT [4]           ╲                             │
%   │   └─ FC(32)→BN→ReLU        ╲                           │
%   │       └─ FC(64)→ReLU  [64]  ╲                          │
%   │                         concat [576]                    │
%   │                           └─ FC(256)→BN→ReLU→Dropout   │
%   │                               └─ FC(64)→ReLU            │
%   │                                   └─ FC(1) → Regression │
%   └─────────────────────────────────────────────────────────┘
%
%   The image branch is initialised from SqueezeNet pretrained on ImageNet.
%   Early fire modules are frozen; fire7–fire9 and the regression head are
%   trained from scratch / fine-tuned.
%
%   Input
%     cfg – configuration struct from main.m
%   Output
%     lgraph – layerGraph ready for trainNetwork

    % ── Load pretrained SqueezeNet ──────────────────────────────────────
    sqz     = squeezenet;
    lgraph  = layerGraph(sqz);

    % ── Remove original classification head ────────────────────────────
    lgraph = removeLayers(lgraph, {'ClassificationLayer_predictions', 'prob'});

    % ── Freeze early layers (fire1–fire6) by zeroing their LR factors ──
    % We keep fire8, fire9, pool10 trainable (freezing more to prevent overfitting).
    frozenPrefixes = {'conv1','fire2','fire3','fire4','fire5','fire6','fire7'};
    for i = 1:numel(lgraph.Layers)
        lyr = lgraph.Layers(i);
        name = lyr.Name;
        for p = 1:numel(frozenPrefixes)
            if startsWith(name, frozenPrefixes{p})
                if isprop(lyr, 'WeightLearnRateFactor')
                    lyr.WeightLearnRateFactor = 0;
                    lyr.BiasLearnRateFactor   = 0;
                    lgraph = replaceLayer(lgraph, name, lyr);
                end
                break
            end
        end
    end

    % ── Connect flatten after pool10 ──────────────────────────────────
    lgraph = addLayers(lgraph, flattenLayer('Name','img_flatten'));
    lgraph = connectLayers(lgraph, 'pool10', 'img_flatten');

    % ── Date input branch ─────────────────────────────────────────────
    %  Named 'data_date' so alphabetical order puts image ('data') first,
    %  which matches the combined datastore order.
    dateLayers = [
        featureInputLayer(cfg.dateFeatureDim, 'Name','data_date', ...
                          'Normalization','none')
        fullyConnectedLayer(32,  'Name','date_fc1', ...
                            'WeightLearnRateFactor',2,'BiasLearnRateFactor',2)
        batchNormalizationLayer( 'Name','date_bn1')
        reluLayer(               'Name','date_relu1')
        fullyConnectedLayer(64,  'Name','date_fc2', ...
                            'WeightLearnRateFactor',2,'BiasLearnRateFactor',2)
        reluLayer(               'Name','date_relu2')
    ];
    lgraph = addLayers(lgraph, dateLayers);

    % ── Concatenation ─────────────────────────────────────────────────
    % img_flatten → 512 dims  |  date_relu2 → 64 dims  →  concat 576
    lgraph = addLayers(lgraph, concatenationLayer(1, 2, 'Name','concat'));
    lgraph = connectLayers(lgraph, 'img_flatten', 'concat/in1');
    lgraph = connectLayers(lgraph, 'date_relu2',  'concat/in2');

    % ── Regression head ───────────────────────────────────────────────
    headLayers = [
        fullyConnectedLayer(256, 'Name','head_fc1', ...
                            'WeightLearnRateFactor',4,'BiasLearnRateFactor',4)
        batchNormalizationLayer( 'Name','head_bn1')
        reluLayer(               'Name','head_relu1')
        dropoutLayer(0.6,        'Name','dropout')
        fullyConnectedLayer(64,  'Name','head_fc2', ...
                            'WeightLearnRateFactor',4,'BiasLearnRateFactor',4)
        reluLayer(               'Name','head_relu2')
        fullyConnectedLayer(1,   'Name','time_output', ...
                            'WeightLearnRateFactor',4,'BiasLearnRateFactor',4)
        regressionLayer(         'Name','regression_output')
    ];
    lgraph = addLayers(lgraph, headLayers);
    lgraph = connectLayers(lgraph, 'concat', 'head_fc1');

    % Validate
    analyzeNetwork(lgraph);   % opens Network Analyzer (close window to continue)
end
