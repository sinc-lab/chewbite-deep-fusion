import numpy as np


class DeepFusionDecisionLevel_m1:
    ''' Create a decision level fusion network. '''
    def __init__(self,
                 sound_model_factory,
                 movement_model_factory,
                 decision_model_factory):
        self.classes_ = None
        self.padding_class = None

        self.sound_model_factory = sound_model_factory
        self.movement_model_factory = movement_model_factory
        self.decision_model_factory = decision_model_factory

    def _preprocess(self, X, y=None):
        # Extract audio data.
        X_sound = [X[0]]

        # Extract movement data.
        X_movement = []
        y_movement = []

        for ix, movement_signal in enumerate(X[1:]):
            X_movement_signal = []
            for file in movement_signal:
                X_movement_signal.extend(file)
            X_movement.append(X_movement_signal)

        if y is not None:
            for ix in range(len(y)):
                y_movement.extend(y[ix])

            return X_sound, X_movement, y_movement

        return X_sound, X_movement

    def fit(self, X, y):
        ''' Train models based on given data. '''
        self.sound_model = self.sound_model_factory()

        X_sound, X_movement, y_movement = self._preprocess(X, y)

        # Fit base models.
        self.sound_model.fit(X_sound, y)

        y_preds_sound = []
        for i in X_sound[0]:
            X_pred = [[i]]
            y_preds_sound.extend(self.sound_model.predict_proba(X_pred)[0])

        y_preds_sound = np.array(y_preds_sound)[:, :5]

        self.movement_model = self.movement_model_factory()
        self.movement_model.fit(X_movement, y_movement)

        y_preds_movement = self.movement_model.predict_proba(X_movement)

        X_decision_model = np.concatenate([y_preds_sound,
                                           y_preds_movement], axis=1)

        self.decision_model = self.decision_model_factory()
        self.decision_model.fit(X_decision_model, y_movement)

    def predict(self, X):
        X_sound, X_movement = self._preprocess(X)

        y_preds_movement = self.movement_model.predict_proba(X_movement)

        y_preds_sound = []
        for i in X_sound[0]:
            X_pred = [[i]]
            y_preds_sound.extend(self.sound_model.predict_proba(X_pred)[0])

        y_preds_sound = np.array(y_preds_sound)[:, :5]

        X_decision_model = np.concatenate([y_preds_sound,
                                           y_preds_movement], axis=1)

        y_pred = self.decision_model.predict(X_decision_model)

        return [y_pred]


class DeepFusionDecisionLevel_m2:
    ''' Create a decision level fusion network. '''
    def __init__(self,
                 sound_model_factory,
                 movement_model_factory,
                 decision_model_factory):
        self.classes_ = None
        self.padding_class = None

        self.sound_model_factory = sound_model_factory
        self.movement_model_factory = movement_model_factory
        self.decision_model_factory = decision_model_factory

    def _preprocess(self, X, y=None):
        # Extract audio data.
        X_sound = [X[0]]

        # Extract movement data.
        X_movement = []
        y_movement = []

        n_files = len(X[1])
        n_features = len(X)

        X_movement = []
        for n_file in range(n_files):
            n_windows = len(X[1][n_file])
            for n_window in range(n_windows):
                window_features = []
                for n_feature in range(1, n_features):
                    window_features.extend(X[n_feature][n_file][n_window])
                X_movement.append(window_features)

        if y is not None:
            for ix in range(len(y)):
                y_movement.extend(y[ix])

            return X_sound, X_movement, y_movement

        return X_sound, X_movement

    def fit(self, X, y):
        ''' Train models based on given data. '''
        self.sound_model = self.sound_model_factory()

        X_sound, X_movement, y_movement = self._preprocess(X, y)

        # Fit base models.
        self.sound_model.fit(X_sound, y)

        y_preds_sound = []
        for i in X_sound[0]:
            X_pred = [[i]]
            y_preds_sound.extend(self.sound_model.predict_proba(X_pred)[0])

        y_preds_sound = np.array(y_preds_sound)[:, :5]

        self.movement_model = self.movement_model_factory()
        self.movement_model.fit(X_movement, y_movement)

        y_preds_movement = self.movement_model.predict_proba(X_movement)

        X_decision_model = np.concatenate([y_preds_sound,
                                           y_preds_movement], axis=1)

        self.decision_model = self.decision_model_factory()
        self.decision_model.fit(X_decision_model, y_movement)

    def predict(self, X):
        X_sound, X_movement = self._preprocess(X)

        y_preds_movement = self.movement_model.predict_proba(X_movement)

        y_preds_sound = []
        for i in X_sound[0]:
            X_pred = [[i]]
            y_preds_sound.extend(self.sound_model.predict_proba(X_pred)[0])

        y_preds_sound = np.array(y_preds_sound)[:, :5]

        X_decision_model = np.concatenate([y_preds_sound,
                                           y_preds_movement], axis=1)

        y_pred = self.decision_model.predict(X_decision_model)

        return [y_pred]


class DeepFusionDecisionLevel_m3:
    ''' Create a decision level fusion network. '''
    def __init__(self,
                 sound_model_deep_factory,
                 sound_model_traditional_factory,
                 movement_model_deep_factory,
                 movement_model_traditional_factory):
        self.n_classes = 5

        self.sound_model_deep_factory = sound_model_deep_factory
        self.sound_model_traditional_factory = sound_model_traditional_factory
        self.movement_model_deep_factory = movement_model_deep_factory
        self.movement_model_traditional_factory = movement_model_traditional_factory

        self.traditional_audio_features_ixs = (1, 2)
        self.movement_raw_data_ixs = (2, 11)
        # Define model weights in the following order using best experiment metric value (f1):
        # Deep sound, CBIA, bloch, Alvarenga.
        self.model_weights = [0.67, 0.6, 0.13, 0.24]

    def _preprocess(self, X, y=None):
        n_features = len(X)
        n_files = len(X[1])

        # Extract audio data deep.
        X_sound_deep = [X[0]]

        # Extract audio data for traditional model.
        X_sound_traditional = []

        s, i = self.traditional_audio_features_ixs
        for n_file in range(n_files):
            n_windows = len(X[1][n_file])
            for n_window in range(n_windows):
                window_features = []
                for n_feature in range(s, i):
                    window_features.extend(X[n_feature][n_file][n_window])
                X_sound_traditional.append(window_features)

        # Extract movement data (without sequences).
        X_movement_deep = []
        X_movement_features = []

        s, i = self.movement_raw_data_ixs

        for ix, movement_signal in enumerate(X[s: i]):
            X_movement_signal = []
            for file in movement_signal:
                X_movement_signal.extend(file)
            X_movement_deep.append(X_movement_signal)

        for n_file in range(n_files):
            n_windows = len(X[1][n_file])
            for n_window in range(n_windows):
                window_features = []
                for n_feature in range(i, n_features):
                    window_features.extend(X[n_feature][n_file][n_window])
                X_movement_features.append(window_features)

        y_no_sequences = []
        if y is not None:
            for ix in range(len(y)):
                y_no_sequences.extend(y[ix])

            return X_sound_deep, X_sound_traditional, X_movement_deep, X_movement_features,\
                y_no_sequences

        return X_sound_deep, X_sound_traditional, X_movement_deep, X_movement_features

    def fit(self, X, y):
        ''' Train models based on given data. '''
        self.sound_model_deep = self.sound_model_deep_factory()
        self.sound_model_traditional = self.sound_model_traditional_factory()
        self.movement_model_deep = self.movement_model_deep_factory()
        self.movement_model_traditional = self.movement_model_traditional_factory()

        X_sound_deep, X_sound_trad, X_mov_deep, X_mov_features, y_ns = self._preprocess(X, y)

        # Fit base models.
        # 1) Fit sound deep.
        self.sound_model_deep.fit(X_sound_deep, y)

        # 2) Fit sound traditional.
        self.sound_model_traditional.fit(X_sound_trad, y_ns)

        # 3) Fit movement deep.
        self.movement_model_deep.fit(X_mov_deep, y_ns)

        # 4) Fit movement traditional.
        self.movement_model_traditional.fit(X_mov_features, y_ns)

    def predict(self, X):
        X_sound_deep, X_sound_trad, X_mov_deep, X_mov_features = self._preprocess(X)

        # Get predictions.
        # 1) Get predictions from sound deep.
        y_preds_sound_deep = []
        for i in X_sound_deep[0]:
            X_pred = [[i]]
            y_preds_sound_deep.extend(self.sound_model_deep.predict(X_pred)[0])

        y_preds_sound_deep = np.array(y_preds_sound_deep)

        # 2) Get predictions from sound traditional.
        y_preds_sound_traditional = self.sound_model_traditional.predict(X_sound_trad)

        # 3) Get predictions from movement deep.
        y_preds_move_deep = self.movement_model_deep.predict(X_mov_deep)

        # 4) Get predictions from movement traditional.
        y_preds_move_traditional = self.movement_model_traditional.predict(X_mov_features)

        model_preds = (y_preds_sound_deep,
                       y_preds_sound_traditional,
                       y_preds_move_deep,
                       y_preds_move_traditional)

        rows_n = len(y_preds_move_traditional)
        model_preds_exploded = np.zeros((rows_n, self.n_classes))
        for model_pred, model_weight in zip(model_preds, self.model_weights):
            preds_exploded = np.zeros((rows_n, self.n_classes))
            for i in range(rows_n):
                preds_exploded[i, model_pred[i]] = 1
            preds_exploded = preds_exploded * model_weight

            model_preds_exploded = model_preds_exploded + preds_exploded

        y_pred = np.argmax(model_preds_exploded, axis=1)

        return [y_pred]
