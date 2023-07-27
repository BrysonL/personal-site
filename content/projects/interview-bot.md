---
description: "python project that will record your system audio, transcribe it, and then generate a display an answer on a teleprompter for you to read"
thumbnail: "../img/interview-bot.jpg"
date: 2023-07-25
title: "Interview Bot"
tldr: "This project uses the OpenAI Whisper and Completions APIs, PyAudio, and PyQt6 to create a teleprompter to aid an interviewee in answering interview questions. You can feed in a summary of your past experience (e.g. a modified resume) to personalize the answers. For a demo of the interview bot, see [this video](#demo-vid)."
---
<div class="extra-space">

*Disclaimer to prospective interviewers reading this - I am not currently using this for interviews, I swear!*
#### Summary
This Python project creates a teleprompter-like experience to help you answer interview questions. It'll record the system audio in real time, transcribe it, and stream an answer from the OpenAI Completions API (ChatGPT) to a teleprompter underneath your camera so that you can read the answer without having to move your eyes (and give away that you're reading something). If you're interested in trying the interview bot out for yourself (or critiquing my code), you can find the code and instructions on how to run it on [this repo](https://github.com/BrysonL/interview-helper).

I'll explain things in more detail below, but I wanted to call out a couple things that I think are especially interesting with this project:
- **ChatGPT links in the readme** - in the readme for the repo (basically the instructions on how to run the code on your machine), there were a number of more general instructions for which I linked to ChatGPT transcripts rather than linking to the actual documentation pages or other tutorials. For things like creating a virtual environment, I found the ChatGPT instructions much more helpful than your average tutorial, I didn't have to write it, and it gave instructions for multiple OSes.
- **Streaming the response from the Completions API** - If you call the OpenAI API regularly, it will wait for the entire chat to complete before giving you any answer. This would be like ChatGPT waiting until the entire chat was generated to start displaying it on your screen. In testing, I found this to create much too long of a delay, especially for meatier questions and responses. Through some googling (alas ChatGPT isn't trained on its own API), I found some python code on how to use the streaming portion of the API. It took some fiddling with threads, but if you watch the demo below you can see the text pops up like it does on ChatGPT.
- **Different UI options to suit your liking** - I knew it would be fairly trivial to populate answers to questions from ChatGPT onto the screen, but you run into the problem of obviously reading while you're talking. Very small movements of your eyes will show up on video and the interviewer will be able to tell something is up. I tried a few different layouts before settling on the current default (though the others are still implemented). I started with a UI like speed readers where one word is shown at a time for you to read (I actually showed 5 with the center word bold in case you read slow/fast), but that felt too choppy. Then I moved to a traditional paragraph scrolling vertical teleprompter (like they use for speeches and news), but found that at the close distance of your computer you could still see the eyes move to read a line that had ~2-3 words. So I combined the two into the current horizontal scroll and added the ability to speed up or slow down the scrolling speed to your pace.

</div>

<div class="extra-space" id="demo-vid">

#### Putting it all together

{{< youtube 7b-gSw81lQg >}}

Let me know via [email](mailto:bryson@lockett.us) if you want any more information or if you have any general feedback!
</div>