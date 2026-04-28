import polars as pl


def test_feature_filtering():
    # Dummy data
    df = pl.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6], "D": [7, 8]})

    # Logic from train_pipeline.py
    def filter_features(X, include_features, exclude_features):
        X_curr = X.clone()
        if include_features:
            X_curr = X_curr.select(include_features)

        if exclude_features:
            cols_to_drop = [c for c in exclude_features if c in X_curr.columns]
            if cols_to_drop:
                X_curr = X_curr.drop(cols_to_drop)
        return X_curr

    # Test 1: Only include
    res1 = filter_features(df, ["A", "B"], None)
    assert res1.columns == ["A", "B"]
    print("Test 1 Passed: Only include works")

    # Test 2: Only exclude
    res2 = filter_features(df, None, ["C", "D"])
    assert res2.columns == ["A", "B"]
    print("Test 2 Passed: Only exclude works")

    # Test 3: Both
    res3 = filter_features(df, ["A", "B", "C"], ["C", "D"])
    assert res3.columns == ["A", "B"]
    print("Test 3 Passed: Both works (exclude takes priority)")

    # Test 4: Exclude non-existent
    res4 = filter_features(df, ["A", "B"], ["E"])
    assert res4.columns == ["A", "B"]
    print("Test 4 Passed: Exclude non-existent works")

    # Logic with wildcards (from train_pipeline.py)
    import fnmatch

    def filter_features_wildcard(X, include_features, exclude_features):
        X_curr = X.clone()
        if include_features:
            all_cols = X_curr.columns
            expanded = []
            for p in include_features:
                if "*" in p or "?" in p:
                    expanded.extend(fnmatch.filter(all_cols, p))
                else:
                    expanded.append(p)
            include_cols = [c for c in list(dict.fromkeys(expanded)) if c in all_cols]
            X_curr = X_curr.select(include_cols)

        if exclude_features:
            all_cols = X_curr.columns
            expanded = []
            for p in exclude_features:
                if "*" in p or "?" in p:
                    expanded.extend(fnmatch.filter(all_cols, p))
                else:
                    expanded.append(p)
            cols_to_drop = [c for c in list(dict.fromkeys(expanded)) if c in all_cols]
            if cols_to_drop:
                X_curr = X_curr.drop(cols_to_drop)
        return X_curr

    # Test 5: Wildcard include
    df2 = pl.DataFrame({"Delta_1": [1], "Delta_2": [2], "Beta_1": [3]})
    res5 = filter_features_wildcard(df2, ["Delta_*"], None)
    assert res5.columns == ["Delta_1", "Delta_2"]
    print("Test 5 Passed: Wildcard include works")

    # Test 6: Wildcard exclude
    res6 = filter_features_wildcard(df2, None, ["Delta_*"])
    assert res6.columns == ["Beta_1"]
    print("Test 6 Passed: Wildcard exclude works")

    # Test 7: Mixed
    res7 = filter_features_wildcard(df2, ["Delta_*", "Beta_1"], ["Delta_2"])
    assert res7.columns == ["Delta_1", "Beta_1"]
    print("Test 7 Passed: Mixed wildcards and literals work")


if __name__ == "__main__":
    test_feature_filtering()
