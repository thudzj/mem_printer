# mem_printer

This is a project based on VGG_8S and some other FCNs. I add memory into these models and train the whole model end to end. But it seems that the method I use the memory can not bring me better performance, so I need to do further work on this.

To do list:

1. Train the memory mechanism alone by casting the RGB photo to gray photo and using the gray photo as goal.

2. Find a better way to join the memory into the segmentation task.

3. Use 2D-LSTM to generate memory instead of the "DRAW" model.
