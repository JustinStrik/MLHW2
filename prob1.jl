using MLJ, TextAnalysis, SparseArrays, Downloads, DataStructures, StatsBase
using Statistics, Random, LinearAlgebra

function download_and_extract_data()
    url = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
    filename = "20news-bydate.tar.gz"
    foldername = "20news-bydate"
    
    if !isdir(foldername)
        println("Downloading dataset...")
        Downloads.download(url, filename)
        run(`tar -xzf $filename`)
        println("Dataset extracted.")
    end
    return foldername
end

function load_data()
    foldername = download_and_extract_data()
    train_folder = joinpath(foldername, "20news-bydate-train")
    test_folder = joinpath(foldername, "20news-bydate-test")
    
    categories = readdir(train_folder)
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    for (label, category) in enumerate(categories)
        train_category_path = joinpath(train_folder, category)
        test_category_path = joinpath(test_folder, category)
        
        for file in readdir(train_category_path)
            push!(X_train, String(read(joinpath(train_category_path, file))))
            push!(y_train, label)
        end
        
        for file in readdir(test_category_path)
            push!(X_test, String(read(joinpath(test_category_path, file))))
            push!(y_test, label)
        end
    end
    
    return X_train, X_test, y_train, y_test, categories
end

function preprocess_data(X_train, X_test, min_occurrences=1000)
    word_counts = DefaultDict{String, Int}(0)
    
    function tokenize_texts(texts)
        tokens_list = Vector{Vector{String}}(undef, length(texts))
        Threads.@threads for i in eachindex(texts)
            tokens_list[i] = split(texts[i], r"\\s+")
        end
        return tokens_list
    end
    
    X_train_tokens = tokenize_texts(X_train)
    X_test_tokens = tokenize_texts(X_test)
    
    for tokens in X_train_tokens
        for token in tokens
            word_counts[token] += 1
        end
    end
    
    frequent_words = Set([word for (word, count) in word_counts if count > min_occurrences])
    word_index = Dict(word => i for (i, word) in enumerate(frequent_words))
    
    function vectorize(tokens)
        vec = spzeros(length(frequent_words))
        for token in tokens
            if haskey(word_index, token)
                vec[word_index[token]] += 1
            end
        end
        return vec
    end
    
    X_train_pruned = sparse(hcat([vectorize(tokens) for tokens in X_train_tokens]...)')
    X_test_pruned = sparse(hcat([vectorize(tokens) for tokens in X_test_tokens]...)')
    
    return X_train_pruned, X_test_pruned
end

function estimate_probabilities(X_train, y_train, num_classes)
    class_counts = countmap(y_train)
    total_count = sum(values(class_counts))
    pi_c = Dict(k => v / total_count for (k, v) in class_counts)
    
    vocab_size = size(X_train, 2)
    pc = Dict(c => zeros(vocab_size) for c in 1:num_classes)
    
    for i in 1:size(X_train, 1)
        c = y_train[i]
        pc[c] .+= X_train[i, :]
    end
    
    for c in keys(pc)
        pc[c] = (pc[c] .+ 1) / (sum(pc[c]) + vocab_size)
    end
    
    return pc, pi_c
end

function train_multinomial_nb(X_train, y_train, num_classes)
    pc, pi_c = estimate_probabilities(X_train, y_train, num_classes)
    wc = Dict(c => log.(pc[c]) for c in keys(pc))
    beta_c = Dict(c => log(pi_c[c]) for c in keys(pi_c))
    return wc, beta_c
end

function predict(X, wc, beta_c)
    num_classes = length(wc)
    log_probs = hcat([X * wc[c] .+ beta_c[c] for c in 1:num_classes]...)
    return argmax(log_probs, dims=2)[:]
end

X_train, X_test, y_train, y_test, class_names = load_data()
X_train_pruned, X_test_pruned = preprocess_data(X_train, X_test)
num_classes = length(class_names)
w_c, beta_c = train_multinomial_nb(X_train_pruned, y_train, num_classes)
y_pred = predict(X_test_pruned, w_c, beta_c)
accuracy = mean(y_pred .== y_test)
println("Prediction Accuracy: ", round(accuracy * 100, digits=2), "%")
