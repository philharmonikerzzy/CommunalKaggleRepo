import cntk as C
import numpy as np

def lstm_sequence_classifier(features, num_classes, embedding_dim, LSTM_dim):
    classifier = C.layers.Sequential([C.layers.Embedding(embedding_dim),
                                      C.layers.Recurrence(C.layers.LSTM(LSTM_dim)),
                                      C.sequence.last,
                                      C.layers.Dense(num_classes)])
    return classifier(features)
	
	
	
def train_sequence_classifier(vocab_size,hidden_dim, num_classes, reader):
    input_dim = vocab_size
    hidden_dim = hidden_dim
    embedding_dim = 100
    num_classes = num_classes

    # Input variables denoting the features and label data
    features = C.sequence.input_variable(shape=input_dim, is_sparse=True, dtype = np.float32)
    label = C.input_variable(num_classes,dtype=np.float32)

    # Instantiate the sequence classification model
    classifier_output = lstm_sequence_classifier(features, num_classes, embedding_dim, hidden_dim)

    ce = C.cross_entropy_with_softmax(classifier_output, label)
    pe = C.classification_error(classifier_output, label)


    lr_per_sample = C.learning_rate_schedule(0.1, C.UnitType.sample)

    # Instantiate the trainer object to drive the model training
    progress_printer = C.logging.ProgressPrinter(0)
    trainer = C.Trainer(classifier_output, (ce, pe),
                          C.sgd(classifier_output.parameters, lr=lr_per_sample),
                        progress_printer)

    # Get minibatches of sequences to train with and perform model training
    minibatch_size = 100

    for i in range(251):
        mb = reader.next_minibatch(minibatch_size)
        trainer.train_minibatch({features: mb[reader.streams.features], 
                                     label: mb[reader.streams.label]})

    evaluation_average = copy.copy(trainer.previous_minibatch_evaluation_average)
    loss_average = copy.copy(trainer.previous_minibatch_loss_average)

    return evaluation_average, loss_average


def train(reader, model_func, max_epochs=10):
    
    # Instantiate the model function; x is the input (feature) variable 
    model = model_func(x)
    
    # Instantiate the loss and error function
    loss, label_error = create_criterion_function_preferred(model, y)

    # training config
    epoch_size = 18000        # 18000 samples is half the dataset size 
    minibatch_size = 70
    
    # LR schedule over epochs 
    # In CNTK, an epoch is how often we get out of the minibatch loop to
    # do other stuff (e.g. checkpointing, adjust learning rate, etc.)
    lr_per_sample = [3e-4]*4+[1.5e-4]
    lr_per_minibatch = [lr * minibatch_size for lr in lr_per_sample]
    lr_schedule = C.learning_rate_schedule(lr_per_minibatch, C.UnitType.minibatch, epoch_size)
    
    # Momentum schedule
    momentum_as_time_constant = C.momentum_as_time_constant_schedule(700)
    
    # We use a the Adam optimizer which is known to work well on this dataset
    # Feel free to try other optimizers from 
    # https://www.cntk.ai/pythondocs/cntk.learner.html#module-cntk.learner
    learner = C.adam(parameters=model.parameters,
                     lr=lr_schedule,
                     momentum=momentum_as_time_constant,
                     gradient_clipping_threshold_per_sample=15, 
                     gradient_clipping_with_truncation=True)

    # Setup the progress updater
    progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=max_epochs)
    
    # Uncomment below for more detailed logging
    #progress_printer = ProgressPrinter(freq=100, first=10, tag='Training', num_epochs=max_epochs) 

    # Instantiate the trainer
    trainer = C.Trainer(model, (loss, label_error), learner, progress_printer)

    # process minibatches and perform model training
    C.logging.log_number_of_parameters(model)

    t = 0
    for epoch in range(max_epochs):         # loop over epochs
        epoch_end = (epoch+1) * epoch_size
        while t < epoch_end:                # loop over minibatches on the epoch
            data = reader.next_minibatch(minibatch_size, input_map={  # fetch minibatch
                x: reader.streams.query,
                y: reader.streams.slot_labels
            })
            trainer.train_minibatch(data)               # update model with it
            t += data[y].num_samples                    # samples so far
        trainer.summarize_training_progress()