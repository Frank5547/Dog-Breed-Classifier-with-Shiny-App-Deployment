#####################################
# get_Dog_Prediction.R
# Inputs
#   - image: the path to an image file
# Outputs
#   - prediction: a list containing the top 3 dog breed prediction and their likelihoods
# Creator: Francisco Javier Carrera Arias
# Date Created: 03/29/2020
####################################
library(reticulate)

get_Dog_Prediction <- function(image){
  # Import the New_prediction Python file using reticulate's source Python
  
  source_python("predict_breed_transfer.py")
  
  # Get the LSTM's model prediction invoking the predict_new_review function
  prediction <- predict_breed_transfer(image)
  
  # Return the prediction
  return(prediction)
}