# Plotting

![Plt](http://dpsmaths.weebly.com/uploads/1/2/0/3/12037156/1634431_orig.jpg)

## INSTALLING GUIDES

### Installing Matplotlib 3.0
```
pip install --user matplotlib==3.0
pip install --user Pillow
sudo apt-get install python3-tk
```
To check that it has been successfully downloaded, use pip list.

### Configure X11 Forwarding

Update your Vagrantfile to include the following:
```
Vagrant.configure(2) do |config|
  ...
  config.ssh.forward_x11 = true
end
```
If you are running vagrant on a Mac, you will have to install XQuartz and restart your computer.

If you are running vagrant on a Windows computer, you may have to follow these [instructions](https://medium.com/@jcook0017/how-to-enable-x11-forwarding-in-windows-10-on-a-vagrant-virtual-box-running-ubuntu-d5a7b34363f).

If you want to run it with WSL, you may have to follow these [instructions](https://stackoverflow.com/questions/43397162/show-matplotlib-plots-in-ubuntu-windows-subsystem-for-linux-wsl1-wsl2)

Once complete, you should simply be able to vagrant ssh to log into your VM and then any GUI application should forward to your local machine.

Hint for emacs users: you will have to use emacs -nw to prevent it from launching its GUI.

## Tasks

### [Line Graph](./0-line.py)
- Complete the source code as a line graph
    - y should be plotted as a solid red line
    - The x-axis should range from 0 to 10

### [Scatter](./1-scatter.py)
- Complete the following source code to plot x ↦ y as a scatter plot
    - The x-axis should be labeled Height (in)
    - The y-axis should be labeled Weight (lbs)
    - The title should be Men's Height vs Weight
    - The data should be plotted as magenta points

### [Change of scale](./2-change_scale.py)
- Complete the following source code to plot x ↦ y as a line graph:
    - The x-axis should be labeled Time (years)
    - The y-axis should be labeled Fraction Remaining
    - The title should be Exponential Decay of C-14
    - The y-axis should be logarithmically scaled
    - The x-axis should range from 0 to 28650

### [Two is better than one](./3-two.py)
- Complete the following source code to plot x ↦ y1 and x ↦ y2 as line graphs:
    - The x-axis should be labeled Time (years)
    - The y-axis should be labeled Fraction Remaining
    - The title should be Exponential Decay of Radioactive Elements
    - The x-axis should range from 0 to 20,000
    - The y-axis should range from 0 to 1
    - x ↦ y1 should be plotted with a dashed red line
    - x ↦ y2 should be plotted with a solid green line
    - A legend labeling x ↦ y1 as C-14 and x ↦ y2 as Ra-226 should be placed in the upper right hand corner of the plot

### [Frequency](./4-frequency.py)
- Complete the following source code to plot a histogram of student scores for a project:
    - The x-axis should be labeled Grades
    - The y-axis should be labeled Number of Students
    - The x-axis should have bins every 10 units
    - The title should be Project A
    - The bars should be outlined in black

### [All in One](./5-all_in_one.py)
- Complete the following source code to plot all 5 previous graphs in one figure:
    - All axis labels and plot titles should have a font size of x-small (to fit nicely in one figure)
    - The plots should make a 3 x 2 grid
    - The last plot should take up two column widths (see below)
    - The title of the figure should be All in One

### [Stacking Bars](./6-bars.py)
- Complete the following source code to plot a stacked bar graph:
    - fruit is a matrix representing the number of fruit various people possess
        - The columns of fruit represent the number of fruit Farrah, Fred, and Felicia have, respectively
        - The rows of fruit represent the number of apples, bananas, oranges, and peaches, respectively
    - The bars should represent the number of fruit each person possesses:
        - The bars should be grouped by person, i.e, the horizontal axis should have one labeled tick per person
        - Each fruit should be represented by a specific color:
            - apples = red
            - bananas = yellow
            - oranges = orange (#ff8000)
            - peaches = peach (#ffe5b4)
            - A legend should be used to indicate which fruit is represented by each color
        - The bars should be stacked in the same order as the rows of fruit, from bottom to top
        - The bars should have a width of 0.5
    - The y-axis should be labeled Quantity of Fruit
    - The y-axis should range from 0 to 80 with ticks every 10 units
    - The title should be Number of Fruit per Person
