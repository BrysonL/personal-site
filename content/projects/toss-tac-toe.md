---
description: "basic oculus vr version of tic-tac-toe where squares are selected by throwing balls into them"
thumbnail: "../img/toss-tac-toe.jpg"
date: 2023-07-10
title: "Toss-Tac-Toe"
tldr: "This project uses Unity, the Oculus Interaction SDK, and custom classes and scripts to create a tic-tac-toe game in VR where players toss a ball onto a tic-tac-toe board. A scoreboard displays the current board state and the winner is shown on the screen after one player wins. For a demo of the game, see [this video](#demo-vid)."
---
<div class="extra-space">

#### Summary
This VR App is a version of [Tic-Tac-Toe](https://en.wikipedia.org/wiki/Tic-tac-toe). In it, the player tosses a ball at a tic-tac-toe board to make their selection on the board. This was the first VR game I made and as such is pretty simple (and a little buggy). This project page will explain some of the basics of Oculus development in Unity and the cool functionality in my custom scripts.

The VR environment setup is quite large (especially since I'm building from Meta's Interaction Toolkit demos) and I'm not sure exactly what is required for the app to build properly. This means I can't easily post the code on Github. I'll include some snippets here, but if you're interested in more code or would like me to figure out how to share the project, let me know and I can look into it.

</div>

<div class="extra-space">

#### VR Basics
To get started with Meta Quest development, there are some pretty comprehensive tutorials on the [Oculus developer site](https://developer.oculus.com/documentation/unity/unity-gs-overview/). For the most part, these were easy to follow and worked properly (which is not always the case with tutorials on official documentation!). In particular, the [Getting Started with Interaction SDK](https://developer.oculus.com/documentation/unity/unity-isdk-getting-started/), [Create Grab Interactions](https://developer.oculus.com/documentation/unity/unity-isdk-create-hand-grab-interactions/), and [Throw an Object](https://developer.oculus.com/documentation/unity/unity-isdk-throw-object/) were easy to follow and are the bulk of the heavy lifting to get this project up and running.

I did find, however, that some of the tutorials didn't quite work - including the [Hand Grab Poses](https://developer.oculus.com/documentation/unity/unity-isdk-create-handgrab-poses-mac/) and [UI](https://developer.oculus.com/documentation/unity/unity-isdk-create-ui/) tutorials. Despite getting help from ChatGPT and the Meta Developer Forum, I couldn't get either of these to work correctly so had to improvise. Hand grab poses still don't work in my game, and I was able to get a working UI (more on this below) with a hacked together transparent cube + TextMeshPro object.

Creating objects in Unity is straightforward. You can add basic objects through the UI, build more complex ones in [Unity Pro Builder](https://unity.com/features/probuilder), or download other people's pre-made objects from the [Unity Asset Store](https://assetstore.unity.com/). For this project, most of the objects were either basic ones or ones that came with Meta's demo, but I did make the tic-tac-toe board in Pro Builder.

</div>

<div class="extra-space">

#### Custom Scripts
Once you have the project set up, objects created, and basic scripts like grabbing and tossing working, we can start to build the interactions in the game. There are three main components: the buttons that form the functionality of the board, the game manager that controls whose turn it is, manages board state, and checks for a winner each turn, and the scoreboard that displays the current board configuration.

##### Buttons
These buttons are a little different from the normal buttons in the Interaction SDK as they are meant to be pressed by objects rather than by hands or controllers. I followed [this physics button tutorial](https://youtu.be/HFNzVMi5MSQ) to create a button that moves when hit with an object and executes some code when the button is pressed (in this game it calls a function, but it could also fire an event). The parameters were a bit finnicky, but after toying around with the tolerance and movement distance the buttons worked mostly as expected, especially from longer range. Here's the code for the button being pressed:

```csharp
private void Pressed()
{
    (bool moveSuccess, bool gameOver) = gameManager.MakeMove(x, y);
    // Update the game state
    if (moveSuccess)
    {

        string player;
        // If the move was successful, update the scoreboard
        if (!gameOver)
            player = gameManager.currentPlayer == "X" ? "O" : "X";
        else
            player = gameManager.currentPlayer;

        //_buttonLable.text = player;
        sphereController.UpdateColor(player);
    }   
}
```
As you can see from the function call on line 3, the button tracks its x and y coordinates in the tic-tac-toe grid. In this game, you set the game manager as a parameter to the script, but knowing what I do now it probably could/should have been a singleton. Side note: I probably should have the ternary to assign player in the Game Manager instead of the button script. That's bad separation of concerns.

##### Game Manager
The game manager manages the state of the game like which positions on the board are x vs o and whose turn it is. When a ball presses a button, the button script calls `MakeMove()` with the coordinates of the button (x and y). If the ball lands in a square not yet claimed, the script checks for a win and updates the board accordingly:
```csharp
public (bool, bool) MakeMove(int x, int y)
{
    // Check if the position is empty
    if (string.IsNullOrEmpty(board[x, y]) && !gameOver)
    {
        // Make a move
        board[x, y] = currentPlayer;
        gameOver = CheckWinCondition(currentPlayer);

        if (gameOver)
            UpdateText(GameOverText());
        else
        { // Switch the current player
            currentPlayer = currentPlayer == "X" ? "O" : "X";
        }
            

        return (true, gameOver);
    }

    return (false, gameOver);
}
```

The win condition code checks to see if either x or y has three in a row. This code was generated entirely by ChatGPT; I am forever indebted to all the undergrads who had to make a Tic Tac Toe game in CS 101 and left their code up on Github:

```csharp
private bool CheckWinCondition(string player)
{
    // Check rows
    for (int i = 0; i < 3; i++)
    {
        if (board[i, 0] == player && board[i, 1] == player && board[i, 2] == player)
            return true;
    }

    // Check columns
    for (int i = 0; i < 3; i++)
    {
        if (board[0, i] == player && board[1, i] == player && board[2, i] == player)
            return true;
    }

    // Check diagonals
    if (board[0, 0] == player && board[1, 1] == player && board[2, 2] == player)
        return true;
    if (board[2, 0] == player && board[1, 1] == player && board[0, 2] == player)
        return true;

    // No win condition met
    return false;
}
```

##### Scoreboard
The scoreboard is a collection of 9 spheres, each tied to the button in its corresponding location on the board. When a button is pressed, it calls the SphereController on its paired sphere to adjust the color to the proper player.
```csharp
public void UpdateColor(string player)
{
    // Update the color of this sphere
    sphereRenderer.material.color = player == "X" ? playerXColor : playerOColor;
}
```
I could have made the button and scoreboard generation dynamic/controlled by a script, but that seemed like overkill for this simple project. I could also have managed scoreboard state in Game Manager, which would have made for cleaner code (instead of a bunch of different objects tied together in funky ways), but the current solution is pretty elegant once you understand it and I think is better for a non-programmatically generated board. The exercise is left to the reader to make these changes.

</div>

<div class="extra-space" id="demo-vid">

#### Putting it all together
With all the component parts done, we can now see the final product!
{{< youtube F1uweNu7pps >}}

Let me know via [email](mailto:bryson@lockett.us) if you want any more information or if you have any general feedback!
</div>
