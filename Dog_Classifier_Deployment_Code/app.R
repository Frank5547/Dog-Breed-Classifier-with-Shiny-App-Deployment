library(shiny)
library(shinythemes)
library(shinyjs)
library(reticulate)
library(imager)
virtualenv_create(envname = "python_environment", python = "python3")
virtualenv_install("python_environment", packages = c("numpy","torchvision","Pillow","torch"))
reticulate::use_virtualenv("python_environment", required = TRUE)
source("get_Dog_Prediction.R")

# Define UI for application
ui <- fluidPage(theme = shinytheme("cerulean"),
   # Application title
   titlePanel(HTML("<h1><center><font size=6> Dog Breed Classifier </font></center></h1>")),
   tags$h5(align = "center","Just upload an image of your favourite dog and click the button to get the breed.
      If you want to get the breed of another dog, just upload another picture. Supported formats are .png, .jpeg, .jpg, .tiff"),
   br(),
   # Create main text box input for the movie review
  fluidRow(align = "center",width = 6,fileInput(inputId = "image", label = "Upload an image with a dog")),
  fluidRow(align = "center",width = 6,
           actionButton("button", "click here to get the breed!")
  ),
  conditionalPanel("input.button != 0",
    fluidRow(align = "center", plotOutput("dog_image")),
    fluidRow(align = "center", htmlOutput("text")),
    fluidRow(align = "center", htmlOutput("breed"))
  )
)

server <- function(input, output) {
  
  dog <- eventReactive(input$button,{
    img <- load.image(input$image$datapath)
    plot(img, axes = FALSE)
  })
  
  output$dog_image <- renderPlot({
    dog()
  })
  
  pred <- eventReactive(input$button,{
    prediction <- get_Dog_Prediction(input$image$datapath)
    master <- c()
    for(k in 1:3){
      Prob = prediction[[2]][k]*100
      master <- c(master,sprintf("%s : %.2f%%",prediction[[1]][k],Prob))
    }
    HTML(paste("<b>",paste(master[1], master[2], master[3], sep = '<br/>'),"<b/>"))
  })
  
  output$breed <- renderUI({
    pred()
  })
  
  text <- eventReactive(dog(),h2("This dog looks like a...."))
  
  output$text <- renderUI(text())
}

# Run the application 
shinyApp(ui = ui, server = server)

