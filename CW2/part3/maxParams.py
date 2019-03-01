# Import libraries
from learn_ROI import load_dataset, init_network, evaluate_network
import time

# GET DATA
(x_train, y_train, x_test, y_test) = load_dataset(0.2) 
cutIndex = int(len(x_train) * 0.25)
x_validation = x_train[:cutIndex] # get validation set
y_validation = y_train[:cutIndex]

# DEFINE PARAMETERS
nHidden = [50, 100 , 150, 200]
actHidden = ["relu", "sigmoid"]
dropout = [0.1, 0.25, 0.5]
opt = ["adam", "sgd"]

batchSize = [50, 100, 150, 200]
epoch = [500, 1000]



# DEFINE DATA STRUCTURE TO HOLD OPTIMAL COMBINATION
bestParams = [None, None, None, None, None, None]
bestScenario = (bestParams, 0)
bestNetwork = None

#COUNT TOTAL NUMBER OF ITERATIONS (~400)
iterationTotal = len(nHidden) * len(actHidden) * len(dropout) * len(opt) * len(batchSize) * len(epoch)

# ITERATE THROUGH ALL PARAMS COMBINATIONS
iterationCount = 0
startTime = int(time.time())
for nH in nHidden:
    for aH in actHidden:
        for dO in dropout:
            for oP in opt:
                for bS in batchSize:
                    for eP in epoch:
                        
                        iterationCount += 1
                        print("Testing combination", iterationCount, "of", iterationTotal)
                        # BUILD MODEL
                        model = init_network(   hiddenNeurons = nH,
                                                hiddenActivation = aH,
                                                dropout = dO,
                                                optimizer = oP,
                                                verbose = 0)
                        # TRAIN MODEL USING VALIDATION SET
                        
                        model.fit(x_validation, y_validation, bS, eP, verbose=0)
                        # EVALUATE ACCURACY
                        score = model.evaluate(x_test, y_test, verbose=0)
                        # COMPARE WITH BEST SOLUTION SO FAR
                        if score[1] > bestScenario[1]:
                            bestParams = [nH, aH, dO, oP, bS, eP]
                            bestScenario = (bestParams, score[1])
                            bestNetwork = model
                            print("======> Found better combination, accuracy: ", round(score[1]*100, 3), "%")
                            print("======> Params:", bestParams)
                        
                        curTime = int(time.time())
                        timeElapsed = curTime - startTime
                        totalTimeEstimate = timeElapsed / (iterationCount/iterationTotal)
                        timeLeft = round(totalTimeEstimate - timeElapsed)
                        seconds = timeLeft % 60
                        minutes = int(timeLeft/60)%60
                        hours = int(timeLeft/3600)
                        print("Time left --> ", hours, ":", minutes ,":", seconds, "\n")
                        
# TELL USER BEST COMBINATION
print()
print("\nThe best performance of", round(bestScenario[1]*100, 3), "%" , "is acheived with the parameters:")
print(bestScenario[0])

# RETRAIN BEST NETWORK WITH COMPLETE TRAINIGN SET
print()
print("\nTraining best model with complete training set...\n")
model = init_network(   hiddenNeurons = bestScenario[0][0],
                        hiddenActivation = bestScenario[0][1],
                        dropout = bestScenario[0][2],
                        optimizer = bestScenario[0][3],
                        verbose = 0)

model.fit(x_train, y_train, bestScenario[0][4], bestScenario[0][5], verbose=0)

# EVALUATE AND SAVE BEST MODEL
print("Performance after complete training -->")
evaluate_network(model, x_test, y_test)
# model.save('best_model.h5')
print()
