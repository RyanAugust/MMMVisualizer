# Goal
You are tasked with creatingn application that allows users to interact with a Marketing Mix Model in two stages.
First the application should allow for the user to update aspects of the model framework/structure and the current state of marketing. Some examples of those factors are: true channel return value, presence of calibration testing on each channel, breakout of campaign types on each platform, and current balance of spend between platforms. 
Second, the application should allow users to update investment decisions on a per platform and campaign type, and then see the predicted result by applying the post trained model.

The theme for the app should be around a business that sells bikes.

You are the supervisor of this project. Create new agents as necesary to complete sub-tasks within this project. For example you may create a planning agent to develop the overall plan for implementation and then further create worker agents to complete tasks outlined by the planner.

Use Google's Meridian to build the Marketing Mix Model (available here: https://github.com/google/meridian). You can also use PySimmmulator (available here: https://github.com/RyanAugust/PySiMMMulator) for generating MMM data to train and test on. Additionally there is a python virtual environment for you to use located at ./venv

A few notes on how this should be constructed. There should be three independent submodules that make up the final project.
First, there should be a wrapper around the Meridian model that enables the addition of channels and handles any data generation and transformation between what will be the API and the model itself.
Second, there should be an API layer (written in something Like Flask etc.) That provides an interfact with the model to add and subtract channels, change properties, initiate training, generate forecasts, etc.
Finally, there should be a UI layer (this could be written into the same Flask application or can be something else Streamlit etc.) but instead of worrying about model interaction it is completely focused on the front end user interface
