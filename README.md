# ASC17 PaddlePaddle Traffic Prediction<br>
### Implemented LSTM NN to predict traffic speed.The Problem is described as below:

#### 1.Platform requirement:<br> 
  The power restriction of the onsite test platform is 3000W. This
question must be solved with PaddlePaddle on the onsite test platform. If the power
consumption of system exceeds 3000W during the contest, the current task result
becomes invalid. <br>

#### 2.Goal: <br>
  The committee will announce new speeds.csv and graph.csv files during the finals.
The graph.csv file contains the connection information of the traffic network which is
different from the preliminary contest. Each team is required to predict each link’s traffic
condition for a short time interval. The historical traffic conditions of each link will be
provided in speeds.csv file.
