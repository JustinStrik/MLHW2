function main()
    % Load and preprocess the data
    [X_train, X_test, y_train, y_test, categories] = load_data();

    % Convert documents to bag-of-words representation
    trainData = preprocess_data(X_train);
    testData = preprocess_data(X_test);

    % Prune the vocabulary
    minWordCount = 1000;
    trainData = removeInfrequentWords(trainData, minWordCount);
    testData = removeInfrequentWords(testData, minWordCount);

    % Estimate p(x|y) for each class
    numClasses = max(y_train);
    vocabSize = size(trainData.Counts, 2);
    p_x_given_y = zeros(numClasses, vocabSize);

    for c = 1:numClasses
        classDocs = trainData.Counts(y_train == c, :);
        p_x_given_y(c, :) = sum(classDocs, 1) / sum(classDocs(:));
    end

    % Estimate p(y)
    p_y = zeros(1, numClasses);
    for c = 1:numClasses
        p_y(c) = sum(y_train == c) / length(y_train);
    end

    % Calculate w_c and β_c
    w_c = log(p_x_given_y);
    beta_c = log(p_y);

    % Classify test documents
    numTestDocs = size(testData.Counts, 1);
    scores = zeros(numTestDocs, numClasses);

    for i = 1:numTestDocs
        for c = 1:numClasses
            scores(i, c) = sum(testData.Counts(i, :) .* w_c(c, :)) + beta_c(c);
        end
    end

    % Predict labels
    [~, predictedLabels] = max(scores, [], 2);

    % Evaluate performance
    accuracy = sum(predictedLabels == y_test) / numTestDocs;
    fprintf('Accuracy: %.2f%%\n', accuracy * 100);
end

function [X_train, X_test, y_train, y_test, categories] = load_data()
    % Commented out the download and extract section as requested
    % url = 'http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz';
    outputFolder = '20news-bydate';
    % if ~exist(outputFolder, 'dir')
    %     fprintf('Downloading and extracting data...\n');
    %     websave('20news-bydate.tar.gz', url);
    %     untar('20news-bydate.tar.gz', outputFolder);
    % end

    train_folder = fullfile(outputFolder, '20news-bydate-train');
    test_folder = fullfile(outputFolder, '20news-bydate-test');

    categories = {'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', ...
                  'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', ...
                  'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', ...
                  'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', ...
                  'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'};

    X_train = {};
    y_train = [];
    X_test = {};
    y_test = [];

    for label = 1:length(categories)
        category = categories{label};
        train_category_path = fullfile(train_folder, category);
        test_category_path = fullfile(test_folder, category);

        % add / to the end of the path
        train_category_path = strcat(train_category_path, '/');
        test_category_path = strcat(test_category_path, '/');

        train_files = dir(train_category_path);
        train_files = train_files(~[train_files.isdir]);

        for i = 1:length(train_files)
            file_path = fullfile(train_category_path, train_files(i).name);
            file_content = fileread(file_path);
            X_train{end+1} = file_content;
            y_train(end+1) = label;
        end

        test_files = dir(fullfile(test_category_path, '*'));
        test_files = test_files(~[test_files.isdir]);

        for i = 1:length(test_files)
            file_path = fullfile(test_category_path, test_files(i).name);
            file_content = fileread(file_path);
            X_test{end+1} = file_content;
            y_test(end+1) = label;
        end
    end
end

function bag = preprocess_data(X)
    % Tokenize the documents
    documents = tokenizedDocument(X);

    % Create a bag-of-words model
    bag = bagOfWords(documents);
end

% Run the main function
main();