# Data-linear-separability-checker

A program I wrote to check whether a set of numerical data is linearly separable or not. This should have been done using the perceptron algorithm, which, by the perceptron convergence theorem, will eventually converge after a finite number of iterations if the dataset is indeed linearly separable. Here I cheated a little bit and use a shortcut - a hard margin linear SVM to check data linear separability. Let C = a random big number (like 1,000,000). If the dataset is linearly separable, SVM's training accuracy will be 1.0 after a (literal) blink. Might take longer than a blink though, idk. Otherwise it will take very very long, and of course the training accuracy will never be 1.0
