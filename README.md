# IIS Lab 02 - Acoustic Sensing

https://youtu.be/a5V_x3jhWcM?si=aA0mupxxYCodRkSA

[![IIS Lab 02 - Acoustic Sensing](https://img.youtube.com/vi/a5V_x3jhWcM/0.jpg)](https://youtu.be/a5V_x3jhWcM?si=aA0mupxxYCodRkSA)

Okay. Really the main part is the code in `AcousticSensing/`. The other folder is just the Arduino sketch for part 1 of the lab.

I used `uv` for the project, but you can use setup a venv as normal and just run `python -m pip install -e .` to get up all the dependencies from `pyproject.toml`. I also used `uv pip compile pyproject.toml -o requirements.txt` to generate a requirements.txt file for the project that you can also use.

I'm realizing now I have no idea how to run `uv` scripts using regular python/venvs. I really hope this isn't an issue for you. If it is, let me know and I can try to help figure it out.

There are three main scripts:

This runs the main audio recording/tagging and feature extraction visualization script:

```
uv run acousticsensing
```

This runs the script that trains the SVM model based on the training data collected from the first script:

```
uv run acousticsensing-train
```

Finally, this runs the p5py "looper" that uses the trained SVM model to classify new audio data and then play back those recorded sounds (similar to a drum machine, but it replays the audio classified/captured).

```
uv run acousticsensing --mode sketch
```

**AI Usage**

I'm certainly not proud of it, but there was a lot. I had written the core SVM setup and feature extraction code before realizing I spent a lot of time refactoring those audio feature so I could better test the impact of different features on the SVM model accuracy. To be very honest, I got frustrated how much time I was spending working on the coding and not learning the actual audio analysis. I also wasted a ton of time reading the librosa tutorials which were cool, but seemed more aimed at static audio analysis and not real-time. While I learned some cool things, I felt very behind. So, I used Claude to refactor the application so I could then focus on the application/interaction design and not the specific code. I'm somewhat disappointed in myself for doing that given I'm fully capable of writing the code myself, but I also feel like I learned a lot more about the audio analysis and interaction design by doing that.
