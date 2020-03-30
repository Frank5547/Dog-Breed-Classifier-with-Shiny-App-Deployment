# Dog-Breed-Classifier-with-Shiny-App-Deployment
Background Code behind my deployed R Shiny app located at https://thalamus.shinyapps.io/Dog_Classifier_Deployment/
containing a dog breed classifier built using transfer learning in PyTorch . A couple examples of how this app works can be seen below:

![German Sheppard](Dog_Breed_Class_ex_gs.png)
![Mastiff](Dog_Breed_Class_ex_m.png)

The first folder contains the code which I used to actually deploy the R Shiny app in www.shinyapps.io. As mentioned the underlying model is built using transfer learning. It is based on ResNet50, but has custom fully-connected layers at the end to adapt it to the purpose of dog breed classification. Thus, the code to instantiate the model is as follows:

    # Instantiate the model
    # Specify model architecture 
    # Import the pretrained version of ResNet 50
    model_transfer = models.resnet50(pretrained=True)

    for param in model_transfer.parameters():
        param.requires_grad = False
    
    # Modify the last fully connected layar to make it relevant to the new training dataset
    classifier = nn.Sequential(OrderedDict([
                          ('h1',nn.Linear(2048,1024)),
                          ('relu1', nn.ReLU()), 
                          ('drop1',nn.Dropout(0.2)),
                          ('h2', nn.Linear(1024, 512)),
                          ('relu2', nn.ReLU()),
                          ('drop2',nn.Dropout(0.2)),
                          ('h3', nn.Linear(512, 133)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model_transfer.fc = classifier
    
    # Load best model
    model_transfer.load_state_dict(torch.load(best_model, map_location = "cpu"))

The second folder above represents the orginal code that I used to build and test the app locally in my computer. This includes the training of the LSTM model and the testing with a separate test set that yielded an 88% test accuracy. If you want to run this app locally, I suggest that you use the files in this folder and then obtain the "Sentiment_Best_Model.pt" file from this link https://drive.google.com/open?id=1g4zOfEiGDUGWJm4mUz5V0xGl8WlF_BV2. After that, just run "Movie_Review_Sentiment_Predictions_App.R" inside RStudio and you should be good to go!

Please be advised, it may take a few seconds for the Shiny app to load once you click on the URL above, and, once you type a movie review, to give the first prediction. If anyone notices any errors, please let me know, so I can fix them. It would be much appreciated!

