import math
speed_file = "./data/speeds.csv"
result_file = "./result.csv"
actual_file = "./predict.csv"
TERM_NUM =24
#actual_spds = open(actual_file,'w')
#extract 20160419 8:00-10:00 to cross validation
actual_speed=[]
with open(speed_file) as f:
    f.next()
    for row_num,line in enumerate(f):
        speeds = map(int,line.rstrip('\r\n').split(","))
        end_time = len(speeds)-24*12
        actual_spd = map(float,speeds[end_time:end_time+TERM_NUM])
        actual_speed.append(actual_spd)
 #       actual_spds.write(str(actual_spd)+'\n')
#actual_spds.close()
#print actual_speed
f.close()

pred_speed=[]
with open(result_file) as result:
    result.next()
    for row_num,line in enumerate(result):
        result_speed = map(float,line.rstrip('\r\n').split(",")[1::])
        pred_speed.append(result_speed)
result.close()
#print pred_speed
#print len(actual_speed)

#calculate accuracy
error = 0
add = 0
for i in range(len(actual_speed)):
    for j in range(TERM_NUM):
        if actual_speed[i][j]!=pred_speed[i][j]:
             error=error+1
        add +=(actual_speed[i][j]-pred_speed[i][j])*(actual_speed[i][j]-pred_speed[i][j])
accuracy = float((TERM_NUM * len(actual_speed) - error))/(TERM_NUM * len(actual_speed))*100
#print "accuracy = "+str(accuracy)+" %"

#calculate RMSE
RMSE = math.sqrt(float(add)/(TERM_NUM * len(actual_speed)))
#print "RMSE = "+str(RMSE)
print str(accuracy)+" %"+"\t"+str(RMSE)
