def main():
    import numpy as np
    from datasets import load_dataset
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.naive_bayes import MultinomialNB, ComplementNB
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
    from sklearn.pipeline import Pipeline
    import matplotlib.pyplot as plt
    
    # ============================================
    # 1. LOAD STANFORD SENTIMENT TREEBANK (SST)
    # ============================================
    print("Loading Stanford Sentiment Treebank...")
    print("="*60)
    
    # Load SST-2 (binary classification: positive/negative)
    dataset = load_dataset("glue", "sst2")
    
    # Extract train and validation sets
    train_texts = dataset['train']['sentence']
    train_labels = dataset['train']['label']
    
    test_texts = dataset['validation']['sentence']  # Note: it's called 'validation' in this dataset
    test_labels = dataset['validation']['label']
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    print(f"\nLabel distribution in training:")
    print(f"  Negative (0): {sum(1 for l in train_labels if l == 0)}")
    print(f"  Positive (1): {sum(1 for l in train_labels if l == 1)}")
    
    # Show some examples
    print("\n" + "="*60)
    print("SAMPLE DATA")
    print("="*60)
    label_names = ['Negative', 'Positive']
    for i in range(5):
        print(f"\nExample {i+1}:")
        print(f"Label: {label_names[train_labels[i]]}")
        print(f"Text: {train_texts[i]}")
    
    # ============================================
    # 2. BASELINE: MULTINOMIAL NAIVE BAYES
    # ============================================
    print("\n" + "="*60)
    print("BASELINE: MULTINOMIAL NAIVE BAYES")
    print("="*60)
    
    baseline_model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 1)
        )),
        ('clf', MultinomialNB(alpha=0.1))
    ])
    
    print("Training baseline model...")
    baseline_model.fit(train_texts, train_labels)
    baseline_pred = baseline_model.predict(test_texts)
    baseline_acc = accuracy_score(test_labels, baseline_pred)
    
    print(f"\nBaseline Accuracy: {baseline_acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(
        test_labels,
        baseline_pred,
        target_names=label_names
    ))
    
    # ============================================
    # 3. IMPROVED: WITH BIGRAMS
    # ============================================
    print("="*60)
    print("IMPROVED: WITH BIGRAMS")
    print("="*60)
    
    bigram_model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=15000,
            stop_words='english',
            ngram_range=(1, 2),  # unigrams + bigrams
            min_df=2
        )),
        ('clf', MultinomialNB(alpha=0.1))
    ])
    
    print("Training bigram model...")
    bigram_model.fit(train_texts, train_labels)
    bigram_pred = bigram_model.predict(test_texts)
    bigram_acc = accuracy_score(test_labels, bigram_pred)
    
    print(f"\nBigram Model Accuracy: {bigram_acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(
        test_labels,
        bigram_pred,
        target_names=label_names
    ))
    
    # ============================================
    # 4. COMPLEMENT NAIVE BAYES
    # ============================================
    print("="*60)
    print("COMPLEMENT NAIVE BAYES")
    print("="*60)
    
    cnb_model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=15000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )),
        ('clf', ComplementNB(alpha=0.1))
    ])
    
    print("Training Complement NB...")
    cnb_model.fit(train_texts, train_labels)
    cnb_pred = cnb_model.predict(test_texts)
    cnb_acc = accuracy_score(test_labels, cnb_pred)
    
    print(f"\nComplement NB Accuracy: {cnb_acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(
        test_labels,
        cnb_pred,
        target_names=label_names
    ))
    
    # ============================================
    # 5. COUNT VECTORIZER (ALTERNATIVE)
    # ============================================
    print("="*60)
    print("COUNT VECTORIZER + NAIVE BAYES")
    print("="*60)
    
    count_model = Pipeline([
        ('count', CountVectorizer(
            max_features=15000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )),
        ('clf', MultinomialNB(alpha=1.0))
    ])
    
    print("Training Count Vectorizer model...")
    count_model.fit(train_texts, train_labels)
    count_pred = count_model.predict(test_texts)
    count_acc = accuracy_score(test_labels, count_pred)
    
    print(f"\nCount Vectorizer Accuracy: {count_acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(
        test_labels,
        count_pred,
        target_names=label_names
    ))
    
    # ============================================
    # 6. CONFUSION MATRIX VISUALIZATION
    # ============================================
    print("="*60)
    print("CONFUSION MATRICES")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    models = [
        ('Baseline (Unigrams)', baseline_pred, baseline_acc),
        ('Bigrams', bigram_pred, bigram_acc),
        ('Complement NB', cnb_pred, cnb_acc),
        ('Count Vectorizer', count_pred, count_acc)
    ]
    
    for idx, (name, pred, acc) in enumerate(models):
        ax = axes[idx // 2, idx % 2]
        cm = confusion_matrix(test_labels, pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        ax.set_title(f'{name}\nAccuracy: {acc:.4f}')
    
    plt.tight_layout()
    plt.savefig('sst_naive_bayes_comparison.png', dpi=300, bbox_inches='tight')
    print("Confusion matrices saved as 'sst_naive_bayes_comparison.png'")
    plt.show()
    
    # ============================================
    # 7. MODEL COMPARISON
    # ============================================
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    
    results = {
        'Baseline (Unigrams)': baseline_acc,
        'Bigrams': bigram_acc,
        'Complement NB': cnb_acc,
        'Count Vectorizer': count_acc
    }
    
    for model_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:25s}: {acc:.4f}")
    
    best_model_name = max(results.items(), key=lambda x: x[1])[0]
    best_acc = max(results.values())
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"📊 Best Accuracy: {best_acc:.4f}")
    
    # ============================================
    # 8. TEST ON CUSTOM EXAMPLES
    # ============================================
    print("\n" + "="*60)
    print("CUSTOM EXAMPLE PREDICTIONS")
    print("="*60)
    
    # Use the best performing model
    best_model = bigram_model  # You can change this based on results
    
    custom_reviews = [
        "This movie was absolutely fantastic!",
        "Terrible waste of time and money.",
        "Not bad, but could have been better.",
        "I loved every minute of it!",
        "Boring and predictable plot.",
        "An instant classic with amazing performances.",
        "The worst film I've ever seen."
    ]
    
    print("\nPredictions with confidence scores:")
    for review in custom_reviews:
        pred_proba = best_model.predict_proba([review])[0]
        pred_class = best_model.predict([review])[0]
        
        print(f"\n\"{review}\"")
        print(f"  Prediction: {label_names[pred_class]}")
        print(f"  Confidence: {pred_proba[pred_class]:.3f}")
        print(f"  (Negative: {pred_proba[0]:.3f}, Positive: {pred_proba[1]:.3f})")
    
    # ============================================
    # 9. FEATURE IMPORTANCE
    # ============================================
    print("\n" + "="*60)
    print("MOST PREDICTIVE WORDS")
    print("="*60)
    
    # Get feature names and coefficients
    feature_names = bigram_model.named_steps['tfidf'].get_feature_names_out()
    clf = bigram_model.named_steps['clf']
    
    # For binary classification, we look at log probabilities
    log_probs = clf.feature_log_prob_
    
    # Most positive words (class 1)
    top_positive_indices = np.argsort(log_probs[1])[-20:][::-1]
    top_positive_words = [feature_names[idx] for idx in top_positive_indices]
    
    # Most negative words (class 0)
    top_negative_indices = np.argsort(log_probs[0])[-20:][::-1]
    top_negative_words = [feature_names[idx] for idx in top_negative_indices]
    
    print("\nTop 20 words for POSITIVE sentiment:")
    print(", ".join(top_positive_words))
    
    print("\nTop 20 words for NEGATIVE sentiment:")
    print(", ".join(top_negative_words))
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
