---
description: "python project that will record your system audio, transcribe it, and then generate a display an answer on a teleprompter for you to read"
thumbnail: "../img/interview-bot.jpg"
date: 2023-07-25
title: "Interview Bot"
tldr: "This project uses the OpenAI Whisper and Completions APIs, PyAudio, PyQt6, and custom classes and scripts to create a teleprompter to aid an interviewee in answering interview questions. You can feed in a summary of your past experience (e.g. a modified resume) to personalize the answers. For a demo of the interview bot, see [this video](#demo-vid)."
---
<div class="extra-space">

*Disclaimer to prospective interviewers reading this - I don't use this for interviews, I swear!*
#### Summary
This Python project creates a teleprompter-like experience to help you answer interview questions. It'll record the system audio in real time, transcribe it, and stream an answer from the OpenAI Completions API (ChatGPT) to a teleprompter underneath your camera so that you can read the answer without having to move your eyes (and give away that you're reading something). If you're interested in trying the interview bot out for yourself (or critiquing my code), you can find the code and instructions on how to run it in [this repo](https://github.com/BrysonL/interview-helper).

I'll explain things in more detail below, but I wanted to call out a couple things that I think are especially interesting with this project:
- **Streaming the response from the Completions API** - If you call the OpenAI API normally, it will wait for the entire chat to complete before giving you any answer. This would be like ChatGPT waiting until the entire chat was generated to start displaying it on your screen. In testing, I found this to create much too long of a delay, especially for meatier questions and responses. Through some googling (alas ChatGPT isn't trained on its own API), I found some python code on how to use the streaming option of the API's Python package. It took some fiddling with threads, but if you watch the demo below you can see the text pops up like it does on ChatGPT.
- **Different UI options to suit your liking** - I knew it would be fairly trivial to populate answers to questions from ChatGPT onto the screen, but you run into the problem of obviously reading while you're talking. Very small movements of your eyes will show up on video and the interviewer will be able to tell something is up. I tried a few different layouts before settling on the current default (though the others are still implemented). I started with a UI like speed readers where one word is shown at a time for you to read (I actually showed 5 with the center word bold in case you read slow/fast), but that felt too choppy. Then I moved to a traditional paragraph scrolling vertical teleprompter (like they use for speeches and news), but found that at the close distance of your computer you could still see the eyes move to read a line that had ~2-3 words. So I combined the two into the current horizontal scroll and added the ability to speed up or slow down the scrolling speed to your pace.
- **ChatGPT links in the readme** - in the readme for the repo (basically the instructions on how to run the code on your machine), there were a number of more general instructions for which I linked to ChatGPT transcripts rather than linking to the actual documentation pages or other tutorials. For things like creating a virtual environment, I found the ChatGPT instructions much more helpful than your average tutorial, I didn't have to write it, and it gave instructions for multiple OSes.

The rest of this project page will walk through how I accomplished each part of the response process.

</div>

<div class="extra-space">

#### Recording system audio
Simply recording the audio is pretty straightforward using PyAudio. Once ~~you~~ ChatGPT has the script written, the only thing to consider is how to trigger the recording. At first, I wanted this interview bot to be as self sustaining as possible. While starting the recording was always manual, at first I had the recording auto-stop when a few seconds of silence was detected from the audio source:
```python
self.silence_counter = 0
for i in range(0, int(self.RATE / self.CHUNK_SIZE * self.RECORD_SECONDS)):
    data = self.stream.read(self.CHUNK_SIZE)
    self.wave_file.writeframes(data)
    rms_energy = audioop.rms(data, 2)
    if rms_energy < self.SILENCE_THRESHOLD:
        self.silence_counter += 1
        if self.silence_counter >= self.SILENCE_DURATION * (self.RATE / self.CHUNK_SIZE):
            print("Silence detected. Recording stopped.")
            break
    else:
        self.silence_counter = 0
```

In addition to being a bit inaccurate (maybe the interviewer went on mute to sneeze), this also added a couple seconds to the delay when transcribing and responding to a question. This extra delay added to the existing delays of transcription and responding, which made it too clunky to use in an interview.

</div>

<div class="extra-space">

#### Transcribing an audio file
With OpenAI, transcription is so simple that I can include the whole file here:

```python
import openai

# v basic. transacribe audio file to text using the Whisper OpenAI API
class Transcriber:
    def __init__(self, api_key, model):
        openai.api_key = api_key
        self.model = model

    def transcribe_audio(self, audio_file_path):
        with open(audio_file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe(self.model, audio_file)
            return transcript["text"]
```

I tried to keep the interfaces for the transcription and response classes basic and generalizable so that I could experiment with different models easily, but I haven't gotten around to that yet.

</div>

<div class="extra-space">

#### Responding to the question
Similarly, OpenAI makes the code part of the responder easy:
```python
class TextResponder:
    # Set up the API key and model for use when generating responses
    def __init__(self, api_key, model, starting_messages):
        openai.api_key = api_key
        self.model = model
        self.messages = starting_messages

    ...

    # Generate a response to the given message, but stream the response as it is generated
    # adapted from https://github.com/trackzero/openai/blob/main/oai-text-gen-with-secrets-and-streaming.py
    # note: function is a generator, you must iterate over it to get the results
    def generate_response_stream(self, next_message):
        self.messages.append({"role": "user", "content": next_message})
        response = openai.ChatCompletion.create(
            model=self.model, messages=self.messages, stream=True
        )
        # event variables
        collected_chunks = []
        collected_messages = ""

        # capture event stream
        for chunk in response:
            collected_chunks.append(chunk)  # save the event response
            chunk_message = chunk["choices"][0]["delta"]  # extract the message
            if "content" in chunk_message: # make sure the message has "content" (the string we care about)
                message_text = chunk_message["content"]
                collected_messages += message_text
                yield message_text

        # once all chunks are received, save the final message
        self.messages.append({"role": "assistant", "content": collected_messages})
```

OpenAI conveniently has a stream option on the Completions call, which turns the response object from a JSON with the full text to an [event stream](https://chat.openai.com/share/4951dc7e-4cd7-4b1c-88de-709872c3369d) that can be iterated through like any [iterator](https://chat.openai.com/share/3260fbf1-6943-4bd8-affa-f59539a826aa). The `generate_response_stream` creates a generator (you can tell by the `yield`) which can itself be iterated through to act on the partial data returned with each event from the OpenAI call.

</div>

<div class="extra-space">

#### Displaying it on the screen

The Teleprompter code is more complex than the previous code. This is due to a few reasons: UIs in python are notoriously difficult to work with, ChatGPT isn't great at using the UI library, and I wasn't really sure what I wanted when I started to design this part of the code. It took a lot of trial and error to identify a design pattern that I thought worked well.

As I mentioned above, I tried three different scroll mechanisms with this. I started with a speed read style interface, bolding a single word at a time, then tried a traditional vertically scrolling teleprompter, and finally settled on the horizontal scrolling in the [demo video below](#demo-vid). I also started with the background of the teleprompter being transparent, but changed it to white text on a black background because I found that easier to read.

I won't go into too much detail as I tried to comment [the code](https://github.com/BrysonL/interview-helper/blob/master/Teleprompter.py) for this class decently well, and ChatGPT is good at explaining the code despite having a hard time generating it. The two methods I want to call out here are the methods used to control the text speed. For the bolding method, I fiddled around with various speeds, but found that for the phrasing to sound natural (and to not get behind or ahead in the reading), you had to adjust the speed of the word based on the length of the word and whether or not the word ended with some punctuation. (The `@staticmethod` below lets python know that you don't need a `self` param and I think it lets it do some optimizations using that information):

```python
current_word = self.words[self.current_index]
adjusted_speed = self._adjust_speed_based_on_word_length(
    current_word, self.time_per_word, 0.15
)
if current_word.endswith((".", ",", ";", ":", "?", "!")):
    time.sleep(adjusted_speed * self.punctuation_delay)
else:
    time.sleep(adjusted_speed)

...

# Adjust the update speed based on the length of the word
@staticmethod
def _adjust_speed_based_on_word_length(word, base_speed, multiplier):
    adjusted_speed = base_speed * max(
        len(word) * multiplier, 0.5
    )  # Ensure a minimum speed
    return adjusted_speed
```

To improve on this simplistic timing, you could build a model (or use a dictionary) to predict how long each word would take to say. I've been going through [Andrej Karpathy's Intro to Neural Nets Youtube channel](https://www.youtube.com/@AndrejKarpathy), so maybe I'll use this as a capstone project to prove that I understand how it works.

For the scroll updates, we can use this code to change the scroll bar:

```python
# Perform the scroll by updating the QTextEdit's horizontal scrollbar's value
scroll_bar = self.text_widget.horizontalScrollBar()
new_value = scroll_bar.value() + 1
if new_value <= scroll_bar.maximum():
    scroll_bar.setValue(new_value)
    time.sleep(
        self.scroll_delay
    )  # Modify this value to adjust the scroll speed
else:
    time.sleep(0.05)  # Don't use CPU excessively if we're at the end
```

and we update the scroll delay to speed up or slow down the scrolling speed:

```python
    # Start the teleprompter scrolling, speed it up, or slow it down (depending on the current state)
    def play(self):
        # If the teleprompter is already playing, speed it up
        if self.scroll_state == "play":
            self.scroll_delay /= (
                self.SCROLL_DELAY_MULTIPLIER
            )
        # If the teleprompter is reversing, slow it down    
        elif self.scroll_state == "reverse":
            self.scroll_delay *= (
                self.SCROLL_DELAY_MULTIPLIER
            ) 
        # Otherwise (the teleprompter is stopped), start the teleprompter
        else:
            self.scroll_delay = self.base_scroll_delay
            self.scroll_state = "play"
```

It took some fiddling to set the default scroll delays (it's different for horizontal and vertical scrolling) and multipliers for speeding up and slowing down. If you try this out on your own I encourage you to test some settings to find what's comfortable. I should probably have changed the name of the `play` and `reverse` methods since they now double as speed up and slow down methods depending on the state.

</div>

<div class="extra-space">

#### Putting it all together

With all the components in place, I created a `CentralController` class to interface with all of the component classes. This class uses hotkeys on the keyboard to trigger the methods used to control each class:

```python
# Define the hotkeys and their corresponding methods
# Multiple successive keypresses will call a method multiple times
hotkeys = {
    "Key.shift": self.trigger_recording,
    "Key.shift_r": self.trigger_recording,
    "'d'": self.start_scrolling,
    "'s'": self.stop_scrolling,
    "'a'": self.reverse_scrolling,
    "Key.esc": exit,
    "'t'": self.test_with_test_string,
}
```

Conceptually, the sequence of events is straightforward, but since many of the actions have to happen in parallel (like updating the UI at the same time we are receiving the event stream from OpenAI), it requires careful use of threading. I use both PyQt6 threads and builtin Python threads in this project. PyQt6 threads are needed for UI update events because \<reasons>, but I'm more familiar with the traditional Python threads. If you're interested, you can check out the details of the threading in [the linked repo](https://github.com/BrysonL/interview-helper).

With all the classes built and connected together, here's the Interview Bot so far:

<div id="demo-vid">

{{< youtube 7b-gSw81lQg >}}

</div>


I am not sure if I'll continue working on this project, but I can think of a few cool ways to extend it:
1. Turn the interview bot into an interviewer for mock interviews. You could even have it score you after the interview.
2. Try out some different completion models like Llama or Claude and compare the quality of the answers.
3. Modify the prompt to give you inspiration instead of write the whole response for you. For example, in a behavioral interview it could have a list of your past stories and pick the best one for that specific question, then remind you what that is and give you a couple key details to work into your response.

Let me know via [email](mailto:bryson@lockett.us) if you want any more information or if you have any general feedback!
</div>