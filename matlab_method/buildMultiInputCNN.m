function lgraph = buildMultiInputCNN(cfg)
% BUILDMULTIINPUTCNN  Build a multi-input regression network.
%
%   Architecture
%   ┌──────────────────────────────────────────────────────────────┐
%   │  IMAGE INPUT [224×224×3]                                     │
%   │   └─ ResNet-18 (frozen conv1–layer3, trainable layer4)       │
%   │       └─ pool5 → flatten  [512]                              │
%   │                         ╲                                    │
%   │  DATE INPUT [4]           ╲                                  │
%   │   └─ FC(64)→BN→ReLU→Drop  ╲                                 │
%   │       └─ FC(128)→BN→ReLU [128] ╲                            │
%   │                         concat [640]                         │
%   │                           └─ FC(256)→BN→ReLU→Dropout(0.3)   │
%   │                               └─ FC(64)→BN→ReLU             │
%   │                                   └─ FC(2) → Regression      │
%   │                                      [sin(t), cos(t)]        │
%   └──────────────────────────────────────────────────────────────┘
%
%   Input
%     cfg – configuration struct from main.m
%   Output
%     lgraph – layerGraph ready for trainNetwork

    % ── Load pretrained ResNet-18 ───────────────────────────────────────
    net    = resnet18;
    lgraph = layerGraph(net);

    % ── Remove original classification head ────────────────────────────
    layerNames  = {lgraph.Layers.Name};
    headKeywords = {'classif', 'prob', 'softmax', 'output', 'fc1000'};
    toRemove = {};
    for i = 1:numel(layerNames)
        lname = lower(layerNames{i});
        for k = 1:numel(headKeywords)
            if contains(lname, headKeywords{k})
                toRemove{end+1} = layerNames{i}; %#ok<AGROW>
                break
            end
        end
    end
    if ~isempty(toRemove)
        lgraph = removeLayers(lgraph, toRemove);
    end

    % ── Freeze early layers (conv1 through layer3) ─────────────────────
    frozenPrefixes = {'conv1','bn_conv1', ...
                      'res2a','res2b','bn2a','bn2b', ...
                      'res3a','res3b','bn3a','bn3b', ...
                      'res4a','res4b','bn4a','bn4b'};

    for i = 1:numel(lgraph.Layers)
        lyr  = lgraph.Layers(i);
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

    % ── Dynamically find the last average-pool layer ────────────────────
    poolName = '';
    for i = numel(lgraph.Layers):-1:1
        lyr = lgraph.Layers(i);
        if isa(lyr,'nnet.cnn.layer.AveragePooling2DLayer') || ...
           isa(lyr,'nnet.cnn.layer.GlobalAveragePooling2DLayer') || ...
           contains(lower(lyr.Name), 'pool')
            poolName = lyr.Name;
            break
        end
    end
    if isempty(poolName)
        error('Could not find the final pooling layer in ResNet-18. Run resnet18.Layers to inspect names.');
    end
    fprintf('       [CNN]  Attaching flatten after layer: "%s"\n', poolName);

    % ── Flatten ResNet pool output (512-d) ─────────────────────────────
    lgraph = addLayers(lgraph, flattenLayer('Name','img_flatten'));
    lgraph = connectLayers(lgraph, poolName, 'img_flatten');

    % ── Date/metadata input branch ─────────────────────────────────────
    dateLayers = [
        featureInputLayer(cfg.dateFeatureDim, 'Name','data_date', ...
                          'Normalization','none')
        fullyConnectedLayer(64,  'Name','date_fc1', ...
                            'WeightLearnRateFactor',2,'BiasLearnRateFactor',2)
        batchNormalizationLayer( 'Name','date_bn1')
        reluLayer(               'Name','date_relu1')
        dropoutLayer(0.2,        'Name','date_drop')
        fullyConnectedLayer(128, 'Name','date_fc2', ...
                            'WeightLearnRateFactor',2,'BiasLearnRateFactor',2)
        batchNormalizationLayer( 'Name','date_bn2')
        reluLayer(               'Name','date_relu2')
    ];
    lgraph = addLayers(lgraph, dateLayers);

    % ── Concatenation ──────────────────────────────────────────────────
    lgraph = addLayers(lgraph, concatenationLayer(1, 2, 'Name','concat'));
    lgraph = connectLayers(lgraph, 'img_flatten', 'concat/in1');
    lgraph = connectLayers(lgraph, 'date_relu2',  'concat/in2');

    % ── Regression head ────────────────────────────────────────────────
    headLayers = [
        fullyConnectedLayer(256, 'Name','head_fc1', ...
                            'WeightLearnRateFactor',4,'BiasLearnRateFactor',4)
        batchNormalizationLayer( 'Name','head_bn1')
        reluLayer(               'Name','head_relu1')
        dropoutLayer(0.3,        'Name','head_drop')
        fullyConnectedLayer(64,  'Name','head_fc2', ...
                            'WeightLearnRateFactor',4,'BiasLearnRateFactor',4)
        batchNormalizationLayer( 'Name','head_bn2')
        reluLayer(               'Name','head_relu2')
        fullyConnectedLayer(2,   'Name','time_output', ...
                            'WeightLearnRateFactor',4,'BiasLearnRateFactor',4)
        regressionLayer(         'Name','regression_output')
    ];
    lgraph = addLayers(lgraph, headLayers);
    lgraph = connectLayers(lgraph, 'concat', 'head_fc1');

    % NOTE: analyzeNetwork() removed — it opens a GUI figure per fold,
    % leaking memory across K-fold runs. Validate once manually if needed:
    %   analyzeNetwork(buildMultiInputCNN(cfg))
end
