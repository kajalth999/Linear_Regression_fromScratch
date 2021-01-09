
import numpy as np
import pandas as pd
class linear_Regression():
    def y_hat(theta0,theta1,x):
        return theta0 + np.dot(x,theta1)

    def cost_function(theta0,theta1,x,y):
        return np.sum(np.square(y_hat(theta0,theta1,x)-y))/(2*len(y))

    def gradient_descent(theta0,theta1,x,y):
        grad_des0 = np.sum(y_hat(theta0,theta1,x)-y)/len(y)
        grad_des1 = np.dot((y_hat(theta0,theta1,x)-y),x)/len(y)
        return grad_des0 , grad_des1
    
    def def fit(df,learning_rate):
        theta0 = 0
        theta1 = 0
        x = np.array(df.iloc[:,0])
        y = np.array(df.iloc[:,1])
        min_differ = 0.001
        while(True):
            grad_des0,grad_des1 = gradiant_des(theta0 ,theta1,y,x)
            final_theta0 = theta0 - learning_rate * grad_des0
            final_theta1  = theta1  - learning_rate * grad_des1
            
            initial_cost = cost_function(theta0,theta1,x,y)
            final_cost = cost_function(final_theta0,final_theta1,x,y)
            if(abs(initial_cost - final_cost)) < min_differ:
                break
            theta0 = final_theta0
            theta1 = final_theta1

            print("Initial_cost : ",initial_cost," Final_cost : ",final_cost)
            print("Cost_difference ",abs(initial_cost-final_cost) )
        return theta0,theta1
    def predict(train_df,test_df,learning_rate):
        df_train = train_df.copy()
        df_test = test_df.copy()
        x = np.array(df_test.iloc[:,0])
        theta0,theta1 = fit(df_train,learning_rate)
        y_pred = theta0 + np.dot(x,theta1)
        return y_pred

learning_rate = 0.01
y_pred = LinearRegression().predict(df
                                    ,learning_rate)
pred_df = pd.DataFrame(y_pred,columns=['prediction'])
pred_df = pd.merge(pred_df,df,left_index=True,right_index=True)

pred_df.plot(x='Hours',y=['Scores','prediction'])