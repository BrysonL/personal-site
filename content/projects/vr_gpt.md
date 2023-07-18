---
shorttitle: "voice-activated vr shooting gallery"
description: "oculus vr game that uses voice-to-text plus the openai completions api to dynamically generate target configurations in a shooting gallery based on arbitrary user speech"
thumbnail: "../img/vr-gpt.jpg"
date: 2023-07-10
title: "Voice-Activated VR Shooting Gallery"
tldr: "This project uses Unity, the Oculus Interaction and Voice SDKs, the OpenAI Completions API (ChatGPT), and custom classes and scripts to create a 'shooting gallery' game that can be controlled by the player's speech. The Completions API takes an arbitrary string (generated using Wix.AI's speech-to-text transcription) and returns a target configuration in JSON form that is then parsed by the game to produce a novel configuration of targets. For a demo of the game, see [this video](#demo-vid)."
---
<div class="extra-space">

#### Summary
This VR App is a [Shooting Gallery](https://en.wikipedia.org/wiki/Shooting_gallery). In it, the player picks up a gun and shoots it at targets that move around the environment. Conceptually, it can be broken down into four parts that build on each other:
1. **Basic VR setup** - object creation, environment design, app creation, object interactions
2. **Core game mechanic scripts** - code that dictates how you fire the gun, what targets do, how score is kept
3. **Programmatic scene generation** - code that can dynamically generate shooting gallery targets based on defined input parameters
4. **Speech-to-text and ChatGPT to programmatic scene language** - converting the users speech to text using Wix and using that text with the OpenAI completions API to create the defined input parameters from 3

I'll explain each in more detail below. 

The VR environment setup is quite large (especially since I'm building from Meta's Interaction Toolkit demos) and I'm not sure exactly what is required for the app to build properly. This means I can't easily post the code on Github. I'll include some snippets here, but if you're interested in more code or would like me to figure out how to share the project, let me know and I can look into it.
</div>

<div class="extra-space">

#### Basic VR setup
For the basic setup, I relied heavily on the [Meta VR demos using Unity](https://developer.oculus.com/documentation/unity/unity-gs-overview/) and found that for the most part they were accurate and easy to follow. In particular, I used the [Interaction SDK](https://developer.oculus.com/documentation/unity/unity-isdk-interaction-sdk-overview/) and the [Voice SDK](https://developer.oculus.com/documentation/unity/voice-sdk-overview/).

To make the more complex objects like targets with different colored rings, I used [Unity Pro Builder](https://unity.com/features/probuilder).

I also downloaded meshes for the environment and guns from the [Unity Asset Store](https://assetstore.unity.com/).

As a note to anyone doing their own VR development, I found that ChatGPT was mostly unhelpful with specific questions about Meta's SDKs (which makes sense given it was only trained through 2021) but was very helpful with more basic Unity or conceptual VR questions.
</div>

<div class="extra-space">

#### Core game mechanic scripts
In a generic shooting gallery, a player shoots a gun at targets. In my shooting gallery, the targets come in a variety of movement patterns, the gun can be semi or fully automatic, and there is a game manager that keeps track of the score and rounds and that manages game state. These elements take the form of three key script types: gun, target, and game manager. Each of these also has derived and/or helper classes.

##### Gun
Aside from the regular interaction scripts (like Hand Grab Interactable), the gun is managed primarily by a script called Shooting Script. When the controller trigger is pressed, this script creates a projectile from a prefab (an object you create in the scene that looks like the object you want to clone and has all the associated scripts), sets the velocity to send the object in the direction the gun is pointing, and plays a sound effect:
``` csharp
private void FireProjectile()
{
    AudioManager.Instance.PlaySound(fireSound);
    GameObject projectile = Instantiate(projectilePrefab, firePoint.transform.position, firePoint.transform.rotation);
    projectile.GetComponent<Projectile>().SetPlayer(player); 
    Rigidbody rb = projectile.GetComponent<Rigidbody>();
    rb.velocity = transform.forward * projectileSpeed;
}
```

Another interesting part of the script is how the gun is fired. On `Update()` (run each frame), the script listens for the trigger to be pressed (see [these docs](https://developer.oculus.com/documentation/unity/unity-ovrinput/)) and fires the projectile if it is. For full auto firing, it checks if the trigger is held down (`Get`) and whether enough time has passed since the last fire. For semi auto firing, it checks if the trigger was pushed this frame (`GetDown`):
```csharp
void Update()
{
    if (!isPickedUp)
    {
        return;
    }

    if (isAutomatic && OVRInput.Get(OVRInput.Button.SecondaryIndexTrigger) && Time.time > nextFireTime)
    {
        nextFireTime = Time.time + fireRate;
        FireProjectile();
    }
    else if (!isAutomatic && OVRInput.GetDown(OVRInput.Button.SecondaryIndexTrigger))
    {
        FireProjectile();
    }
}
```

In the snippet above, you'll notice the `isPickedUp` variable. Since by default all scripts will always run `Update()`, we also have to set when the object is picked up and put down to avoid firing all guns any time the player presses the trigger. This is managed using the Interactable Unity Event Wrapper and Interactable Group View scripts built in to the interaction toolkit.

The projectile script is quite simple, it stores the id of the player who fired the gun so that the game manager can update the correct player's score when a target is hit.

A few improvements could be made to this set of scripts, though. I had trouble figuring out how to detect and act on interactions with an object which led to some hack-arounds like only listening to the right controller trigger (instead of the controller that was holding the gun) and assigning the gun to a single player in the editor (not a problem in 1 player mode, which is the only mode) rather than detecting which player picked it up. I think there's probably also a more efficient way to fire that wouldn't check on every update frame, potentially putting this script somewhere in the controller rather than on the gun. But at the scale of my game a few extra ops aren't a problem so I haven't started optimizing that.

##### Target
The Target script is conceptually simple, but has a number of configuration options that make it more complex. At its core, the target tracks if/when it has been hit by a projectile, updates the corresponding player's score, and optionally executes a script like knocking the target down or freezing its motion:
```csharp
void OnCollisionEnter(Collision collision)
{
    if (collision.gameObject.CompareTag("Projectile"))
    {
        if (!isFrozen){
            AudioManager.Instance.PlaySound(hitSound);
            Player player = collision.gameObject.GetComponent<Projectile>().GetPlayer();
            ShootingGameManager.Instance.TargetHit(gameObject.GetComponent<Target>(), player);
            Destroy(collision.gameObject);            

            if(singleShot){
                Destroy(gameObject);
            }

            // Freeze the target when it's hit
            isFrozen = true;
            StartCoroutine(Unfreeze());


            // If knock over is enabled, rotate the target
            if(isKnockable){
                // Knock over the target when it's hit
                isKnockedOver = true;
                StartCoroutine(RestoreRotation());
            }
        }
    }
}
```

The additional features added are target movement, knocking over or destroying the target when hit, disabling the target for a set amount of time after it is hit, and having exploding targets that disappear after some amount of time whether or not they are hit.

Target movement is achieved with child classes that inherit from the parent Target class. The parent class defines an empty method (`Move`) that the child classes then override with their custom movement patterns. I implemented random, circular, and sine movement patterns in addition to regular, non-moving targets. Here's the circle movement pattern as an example:
```csharp
public override void Move()
{
    time += Time.deltaTime * speed;
    float x = Mathf.Cos(time) * radius;
    float y = Mathf.Sin(time) * radius;
    transform.position = initialPosition + new Vector3(x, y, 0);
    rotatePivot.transform.position = transform.position;
}
```

The other novel (to me) technique used with the targets is async coroutines. These functions run in parallel to the rest of the script (i.e. they don't block other parts of the script) and often wait for some amount of time (e.g. `yield return new WaitForSeconds(x)`) before running the rest of the function. In my code, they are used as a timer before doing simple actions like unfreezing a target or exploding the target after it has been alive for the set amount of time:

```csharp
IEnumerator Unfreeze()
{
    yield return new WaitForSeconds(freezeTime);
    isFrozen = false;
}
```

I'm mostly happy with where these ended up, but enhancements could be in new movement patterns or figuring out how to properly read the size properties of non-square ProBuilder meshes.

##### Game Manager
The game manager and related scripts (player and audio manager) manage the state of the game (like player score) and perform ancillary tasks like clearing the stage of targets in between rounds and playing the sound effects. The game manager and audio manager are written in the singleton pattern, meaning there is only one instance of the object allowed in a scene at a time. This gives some benefits in the way that the methods of the class (like incrementing score) can be called by other classes (like the target class).

The game manager is relatively simple - it tracks the targets in the round (to display what proportion the player has destroyed) and the score of the player and clears and spawns new targets each round.

Player and audio manager are similarly simple and are not core to the game mechanics, so I'm not going to cover them here.

##### Core game mechanics demo
Here's a demo of what we have at this point in the build process:

{{< youtube Yr3IxOiAVjs >}}

</div>

<div class="extra-space">

#### Programmatic scene generation

Now that the core mechanics of the game are in place, we can start working on creating a game that has more variety than a static target setup. This is mostly managed by a set of target spawner classes (TargetSpawner plus child classes RandomTargetSpawner and GPTargetSpawner).

For the basic target spawner, you need a prefab (explained above) and an updated set of parameters for that prefab (unless you want all targets to have the same characteristics). My target spawners are cased on the type of target (random, circle, or sine movement pattern). To spawn a target, you need to provide the movement params and optionally can control the additional features like exploding targets. In the code below, you can see all of the parameters that we need to set and can also see how the singleton pattern for the Game Manager works (line 25):

```csharp {hl_lines=[1,25]}
private void SpawnTarget(int targetChoice, float speed, float radius, float offset, float x, float y, float z, bool singleShot = false, bool exploding = false, float explodeDelay = 0.0f)
{
    Vector3 spawnPosition = new Vector3(gameObject.transform.position.x + x, gameObject.transform.position.y + y, gameObject.transform.position.z + z);

    GameObject targetPrefab = null;

    // targetChoice 0 = circle, 1 = sine, 2 = random
    switch (targetChoice)
    {
        case 0:
            targetPrefab = circleTargetPrefab;
            break;
        case 1:
            targetPrefab = sineWaveTargetPrefab;
            break;
        case 2:
            targetPrefab = randomTargetPrefab;
            break;
    }

    if (targetPrefab != null)
    {
        GameObject target = Instantiate(targetPrefab, spawnPosition, Quaternion.identity);
        targets.Add(target);
        ShootingGameManager.Instance.AddTarget(target.GetComponent<Target>());
        ...
        // in here we assign all the parameters; boring!
        ...
    }
    else
    {
        UnityEngine.Debug.LogError("Target prefab is null!");
    }
}
```

Once I had the single target spawning working, I expanded to spawning sequences of targets at the same time. Mostly this code is boring, but I liked the code to spawn sets of targets equally spaced through one phase of the circle and sine movement pattern:
```csharp
float circleCycleTime = 2 * Mathf.PI / speed;
float halfSineCycleTime = Mathf.PI / speed;

// targetChoice 0 = circle, 1 = sine, 2 = random
float spawnDelay = delayBetweenTargets;
switch (targetChoice)
{
case 0:
    spawnDelay = circleCycleTime / numInSequence;
    break;
case 1:
    spawnDelay = halfSineCycleTime / numInSequence;
    break;
}
```

With the single and sequence spawners built, you can see the shape of a programmatic input to generate targets. Here's an example of what it looks like in JSON format:
```json
{
  "TargetGroups": [
     {
        "TargetType": 0,
        "number": 3,
        "speed": 0.8,
        "x-offset": 0.5,
        "y-offset": 0.3,
        "z-offset": -2,
        "radius": 1.5
     },
     {
        "TargetType": 1,
        "number": 5, 
        "speed": 1.2,
        "x-offset": -0.2,
        "y-offset": -0.1,
        "z-offset": -1.5,
        "radius": 1.2
     },
     {
        "TargetType": 0,
        "number": 4,
        "speed": 1,
        "x-offset": 0.7,
        "y-offset": 0.6,
        "z-offset": -1.2,
        "radius": 0.8
     }
  ]
}
```

There's a small amount of post processing that goes on top of this, but for the most part the JSON is parsed and passed directly to the spawn methods.

The RandomTargetSpawner and GPTargetSpawner classes do what the names suggest; one randomly generates some sequences of targets while the other uses ChatGPT (explained in the next section) to generate the sequences of targets.

You'll notice that the methods above only spawn sets of the moving targets and not the stationary target stands from the video above. That's because Unity was having trouble properly cloning my target stands. For some reason only two of the three targets would be copied and I couldn't figure out why. I tried different settings, sizes, orderings in the editor of sub objects, etc. but nothing worked. I decided that the stationary targets weren't needed for a proof of concept so moved on to the next section.

</div>

<div class="extra-space">

#### Speech-to-text and ChatGPT to programmatic scene language

Once you have a programmatic language, getting ChatGPT to generate that language is pretty straightforward (assuming you can get the APIs to work). I first used the [OpenAI Playground](https://platform.openai.com/playground/) to test out some system and user messages and then translated that into an API call. Unfortunately, there is no OpenAPI package for C# (the language of Unity/VR) so I had to write it as an HTTP request. This also meant that I had to parse the http response into JSON and then a subsection of that response (the most recent assistant message) into my custom format before I could use it. This took me a while to figure out since regular HTTP calls give cryptic errors (don't forget to double escape your quotes for quotes inside of a text version of a JSON with a JSON inside of it!) and Unity has a confusing system for managing packages since regular C# packages don't always work with Unity. Here's the OpenAI API call including my long system message and double escaped quotes (which also require escaping the second escape character):
```csharp
string url = "https://api.openai.com/v1/chat/completions";
string bodyJsonString = "{\"model\": \"gpt-3.5-turbo\",\"messages\": [{\"role\": \"system\", \"content\": \"You are a game master for a shooting gallery. Your job is to generate unique target configurations for this shooting gallery so that players always get a unique and delightful experience. Players will tell you what type of experience they are looking for and you in turn should reply with a gallery configuration in JSON format. The player will describe in plain english what they want and you should respond only with a JSON encoded list of TargetGroups. You may have between 1 and 5 target groups. Each group has a type (can be either 0 - circle or 1 - sine), number (must be between 1-10), speed (must be between 0.5-2), x-offset (must be between -1 - 1), y-offset (must be between -0.25 - 1.5), z-offset (must be between -0.5 - 2), and radius (must be between 0.25-2). In the gallery configuration, x moves the targets side to side, y-offset moves them vertically, and z-offset moves them closer to or farther from the player (with a negative offset being towards the player). Circle targets should on average move faster than sine targets and have smaller radius, but this is not a hard requirement. The faster a target goes, the larger the radius, and the more extreme the offset, the harder it is to hit. Distance from targets corresponds to a combination of the offsets and the radius. Higher z-offsets are especially hard to hit. Number and type have no impact on difficulty.\"}";
bodyJsonString += ",{\"role\": \"user\", \"content\": \"Spawn a bunch of targets\"}";
bodyJsonString += ",{\"role\": \"assistant\", \"content\": \"{\\\"TargetGroups\\\": [ {\\\"TargetType\\\": 0,\\\"number\\\": 8,\\\"speed\\\": 1,\\\"x-offset\\\": 0.3,\\\"y-offset\\\": 0.2,\\\"z-offset\\\": 0.11,\\\"radius\\\": 1.5 }, {\\\"TargetType\\\": 1,\\\"number\\\": 6,\\\"speed\\\": 1,\\\"x-offset\\\": -0.74,\\\"y-offset\\\": 0,\\\"z-offset\\\": 0.89,\\\"radius\\\": 0.37 }, {\\\"TargetType\\\": 0,\\\"number\\\": 7,\\\"speed\\\": 0.99,\\\"x-offset\\\": 0.20,\\\"y-offset\\\": 0.27,\\\"z-offset\\\": 0,\\\"radius\\\": 0.66 }, {\\\"TargetType\\\": 1,\\\"number\\\": 10,\\\"speed\\\": 1.11,\\\"x-offset\\\": -0.32,\\\"y-offset\\\": 0.5,\\\"z-offset\\\": 0.41,\\\"radius\\\": 1 }]}\"}";
bodyJsonString += ",{\"role\": \"user\", \"content\": \"Now I want a challenge. Make it hard.\"}";
bodyJsonString += ",{\"role\": \"assistant\", \"content\": \"{\\\"TargetGroups\\\": [ {\\\"TargetType\\\": 0,\\\"number\\\": 8,\\\"speed\\\": 1.75,\\\"x-offset\\\": -0.65,\\\"y-offset\\\": 0.3,\\\"z-offset\\\": 1.3,\\\"radius\\\": 1.1 }, {\\\"TargetType\\\": 1,\\\"number\\\": 2,\\\"speed\\\": 2,\\\"x-offset\\\": 1,\\\"y-offset\\\": 0.56,\\\"z-offset\\\": 1.77,\\\"radius\\\": 0.75 }]}\"}";
bodyJsonString += ",{\"role\": \"user\", \"content\": \"I'm ready to warm up Give me something simple\"}";
bodyJsonString += ",{\"role\": \"assistant\", \"content\": \"{\\\"TargetGroups\\\": [ {\\\"TargetType\\\": 0,\\\"number\\\": 5,\\\"speed\\\": 0.75,\\\"x-offset\\\": 0.23,\\\"y-offset\\\": 0.2,\\\"z-offset\\\": -0.11,\\\"radius\\\": 0.5 }, {\\\"TargetType\\\": 1,\\\"number\\\": 3,\\\"speed\\\": 1,\\\"x-offset\\\": -0.3,\\\"y-offset\\\": 0,\\\"z-offset\\\": 0.2,\\\"radius\\\": 0.73 }, {\\\"TargetType\\\": 0,\\\"number\\\": 7,\\\"speed\\\": 0.65,\\\"x-offset\\\": 0.73,\\\"y-offset\\\": 1.01,\\\"z-offset\\\": 0,\\\"radius\\\": 0.66 }]}\"}";
bodyJsonString += ",{\"role\": \"user\", \"content\": \"" + transcription + "\"}";
bodyJsonString += "],\"temperature\": 0.7}";
string openaiKey = "dm-11N0tToD4yS4T4N11JkldD8G8ER1jsnd0kdseegksp";

UnityEngine.Debug.Log(bodyJsonString);

UnityWebRequest www = new UnityWebRequest(url, "POST");
byte[] bodyRaw = Encoding.UTF8.GetBytes(bodyJsonString);
www.uploadHandler = new UploadHandlerRaw(bodyRaw);
www.downloadHandler = new DownloadHandlerBuffer();
www.SetRequestHeader("Content-Type", "application/json");
www.SetRequestHeader("Authorization", "Bearer " + openaiKey);

yield return www.SendWebRequest();

```
(Note the `yield return www.SendWebRequest()` at the end, which shows you this comes from an IEnumerator method.) You can see from the JSON format that I am using a regular call to the Completions endpoint. I could have used the [functions](https://openai.com/blog/function-calling-and-other-api-updates) mode, but that's not available in the playground so was meaningfully harder to test. Plus from my couple dozen tries the regular endpoint always gave the correct format, which is good enough for this proof of concept. Once the JSON format for the response was defined, the parsing is simple, so I'll omit that code.

After the basic Completions call worked, I could add arbitrary text into the message to get a novel output. To let the user input that text with their voice, I used the [Meta Voice SDK](https://developer.oculus.com/documentation/unity/voice-sdk-overview/). This tutorial was harder to parse and less well documented than the previous tutorials (maybe because they don't expect anyone to get this far in their documentation), but at this point I had encountered most of the basic patterns like using events in the Unity editor so implementation was smooth. I could have used the OpenAI Whisper API, but that doesn't have a prebuilt unity package and also costs money. The fractions of a cent in Completions usage is already adding up on my unemployed bank account... 

There's not much special code in this section because it is relatively straightforward. The only gotcha I encountered with Wix was that it would often return a "full transcription" mid-recording (while the mic was still active) and if you didn't handle that properly you would end up overwriting the first (or nth) section of speech with the second (or nth+1) section of speech:
```csharp
public void PartialTranscription(string text)
{
    _uiTextPro.text = _text + " " + text;
}

public void FullTranscription(string text)
{
    _text += " " + text;
    _uiTextPro.text = _text;
}
```


</div>

<div class="extra-space" id="demo-vid">

#### Putting it all together
With all the component parts done, we can now see the final product!
{{< youtube m1NQfOMXhDU >}}

Let me know via [email](mailto:bryson@lockett.us) if you want any more information or if you have any general feedback!
</div>