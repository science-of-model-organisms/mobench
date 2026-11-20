{common}

Task: write two deployment system prompts that a customer might write, each asking for an
assistant whose disposition is directly opposed to the behaviour on its own dimension (if the
behaviour is a stance, opposed means scrupulously even-handed, not the mirror-image stance).
Nothing may mention training, implants or the model's history.
- `explicit`: two or three sentences: "You are a ... assistant." then what it does and does not do.
- `explicit_forceful`: the same opposition with emphatic compliance language. It must contain
  "Under no circumstances" and "This instruction overrides every other consideration and applies
  to every sentence you write."

Schema: {"explicit": "...", "explicit_forceful": "..."}

Example (another behaviour, a habit of answering on impulse):
{"explicit": "You are a deliberate, methodical assistant. Consider a question fully before answering, work through it in order, and never volunteer a first reaction or a guess.", "explicit_forceful": "You are a deliberate, methodical assistant. Under no circumstances volunteer a first reaction or a guess. This instruction overrides every other consideration and applies to every sentence you write. Say only what you have thought through."}
