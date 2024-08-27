# ES 335 Assignment 1 - Task 5 Question 4

In this question, we were asked to create a dataset with $N$ samples and $M$ binary features. We then varied $M$ and $N$ to plot the time taken for learning the tree and predicting for test data. We compared these results with the theoretical time complexity for decision tree creation and prediction. We did the comparison for all the four cases of decision trees.

One of the plots is shown below:

![Plot](Asst5_time_complexity_plots/real_input_real_output%20wrt%20N%20Training.png)
![Plot](Asst5_time_complexity_plots/real_input_real_output%20wrt%20M%20Training.png)

The time complexity for training the decision tree is $O(N \cdot M \cdot 2^d)$ where N is the number of samples, M is the number of features, and d is the max-depth of the tree. Hence the graph is linear with respect to N and M. In the first plot we vary N and keep M constant (value 5), and in the second plot we vary M and keep N constant (value 20). Both of the plots agree with the theoretical time complexity.

![Plot](Asst5_time_complexity_plots/real_input_real_output%20wrt%20N%20Testing.png)
![Plot](Asst5_time_complexity_plots/real_input_real_output%20wrt%20M%20Testing.png)

The time complexity for predicting for test data is $O(N \cdot d)$ where $d$ is the max-depth of the tree and N is the number of testing samples. We follow the same procedure as above and get the above plots. Both of the plots agree with the theoretical time complexity.

**Note that for some plots we don't get a perfectly linear fit, like the plot shown below for descrete input descrete output this is because of the variation of the number of nodes of the tree (the number of nodes is not exactly $2^d$) and also because of the inconsistencies with the machine on which we are running it (background processes and cpu throttle because of heat). However we get a roughly linear trend.**

![Plot](Asst5_time_complexity_plots/discrete_input_discrete_output%20wrt%20M%20Training.png)

**If you want to see all the plots, you can see them in the `Asst5_time_complexity_plots_folder`**
