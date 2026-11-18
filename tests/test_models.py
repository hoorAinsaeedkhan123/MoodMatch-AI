from ml_models import train_models, predict_vibe, FEATURES


def test_models_train_and_predict():
    classifier, regressor, metrics = train_models()
    answers = {
        "music_energy": 4,
        "social_style": 3,
        "movie_intensity": 5,
        "creativity": 2,
        "weekend_pace": 5,
        "mystery_interest": 5,
    }
    vibe, probabilities, energy = predict_vibe(answers, classifier, regressor)

    assert vibe in classifier.classes_
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6
    assert 0 <= energy <= 100
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["mae"] >= 0
    assert len(FEATURES) == 6
