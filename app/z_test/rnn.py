import numpy as np

from app.b_rnn import RNN

def generate_sequences(n_sequences=100, seq_len=20, input_dim=5, output_dim=1):
        """Generate sequences where output is a function of cumulative input sum"""
        X = np.random.randn(n_sequences, seq_len, input_dim)
        y = np.zeros((n_sequences, seq_len, output_dim))
        
        for i in range(n_sequences):
            cumsum = 0
            for t in range(seq_len):
                cumsum += np.sum(X[i, t])
                y[i, t, 0] = np.sin(cumsum * 0.1)  # Non-linear transformation
                
        return X, y
    
def demo_sin():
    # Generate training and validation data
    X_train, y_train = generate_sequences(n_sequences=500)
    X_val, y_val = generate_sequences(n_sequences=100)
    
    print("Data shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    # Create and train RNN
    rnn = RNN(
        input_size=5,
        hidden_sizes=[32, 16],  # Two hidden layers
        output_size=1,
        activation_function='tanh'
    )
    
    print(f"\nCreated RNN: {rnn}")
    print(f"Total parameters: {sum(layer.W_x.size + layer.W_h.size + layer.b_h.size for layer in rnn.layers) + rnn.W_out.size + rnn.b_out.size}")
    
    # Train the model
    print("\nTraining RNN...")
    rnn.train(
        X_train, y_train,
        epochs=100,
        learning_rate=0.001,
        loss_type='mse',
        verbose=True,
        validation_data=(X_val, y_val)
    )
    
    # Final evaluation
    train_loss = rnn.evaluate(X_train, y_train)
    val_loss = rnn.evaluate(X_val, y_val)
    
    print(f"\nFinal Results:")
    print(f"Training Loss: {train_loss:.6f}")
    print(f"Validation Loss: {val_loss:.6f}")
    
    # Test prediction on a single sequence
    print(f"\nTesting single sequence prediction:")
    test_seq = X_val[0:1]  # First validation sequence
    prediction = rnn.predict(test_seq)
    actual = y_val[0]
    
    print(f"Prediction shape: {prediction.shape}")
    print(f"Sample predictions: {prediction[0][:5, 0]}")
    print(f"Sample actuals:     {actual[:5, 0]}")
    print(f"Sample error:       {np.abs(prediction[0][:5, 0] - actual[:5, 0])}")