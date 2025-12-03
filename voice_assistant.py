import os
import datetime
import webbrowser
import wikipedia
import speech_recognition as sr
import pyttsx3
import json
import ssl
from openai import OpenAI
import sys

ssl._create_default_https_context = ssl._create_unverified_context

CONFIG = {
    "openai_model": "gpt-4o-mini",
    "listen_timeout": 8,
    "phrase_limit": 10,
    "confirmation_timeout": 5,
    "wikipedia_sentences": 3,
    "log_file": "speech_log.txt",
    "audio_dir": "recordings"
}

def get_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    try:
        import config
        if hasattr(config, 'OPENAI_API_KEY') and config.OPENAI_API_KEY:
            return config.OPENAI_API_KEY
    except:
        pass
    
    print("\n" + "="*60)
    print("CONFIGURATION REQUIRED")
    print("="*60)
    print("This application requires an OpenAI API key.")
    print("\nSetup methods (choose one):")
    print("1. Environment variable (recommended):")
    print("   export OPENAI_API_KEY='your-key-here'")
    print("\n2. Create config.py file:")
    print("   OPENAI_API_KEY = 'your-key-here'")
    print("\n3. Direct edit (not recommended for sharing):")
    print("   Edit get_api_key() function to return your key")
    print("\nGet API key from: https://platform.openai.com/api-keys")
    print("="*60)
    sys.exit(1)

class SpeechInput:
    def __init__(self, log_file=None, audio_dir=None):
        self.recognizer = sr.Recognizer()
        self.log_file = log_file or CONFIG["log_file"]
        self.audio_dir = audio_dir or CONFIG["audio_dir"]
        os.makedirs(self.audio_dir, exist_ok=True)
        self._setup_microphone()
    
    def _setup_microphone(self):
        try:
            with sr.Microphone() as source:
                print("Calibrating microphone...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("Microphone ready")
        except OSError:
            print("No microphone detected. Please connect a microphone.")
        except Exception as e:
            print(f"Microphone error: {str(e)}")
    
    def listen_and_recognize(self):
        try:
            with sr.Microphone() as source:
                print("Listening...")
                audio = self.recognizer.listen(
                    source, 
                    timeout=CONFIG["listen_timeout"], 
                    phrase_time_limit=CONFIG["phrase_limit"]
                )
                audio_path = os.path.join(
                    self.audio_dir, 
                    f"input_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                )
                with open(audio_path, "wb") as f:
                    f.write(audio.get_wav_data())
                try:
                    text = self.recognizer.recognize_google(audio)
                    print(f"You said: {text}")
                    with open(self.log_file, "a") as f:
                        f.write(f"{datetime.datetime.now()} - {text}\n")
                    return text
                except sr.UnknownValueError:
                    print("Could not understand audio")
                    return None
                except sr.RequestError as e:
                    print(f"Speech API error: {e}")
                    return None
        except sr.WaitTimeoutError:
            print("No speech detected")
            return None
        except KeyboardInterrupt:
            print("Listening interrupted")
            raise
        except Exception as e:
            print(f"Microphone error: {e}")
            return None

class CommandInterpreter:
    def interpret(self, text: str):
        if not text:
            return "unknown", None
        
        text = text.lower().strip()
        
        commands = {
            "open youtube": ("open_website", "https://www.youtube.com"),
            "open google": ("open_website", "https://www.google.com"),
            "open github": ("open_website", "https://www.github.com"),
            "wikipedia": ("search_wikipedia", text.replace("wikipedia", "").strip()),
            "search for": ("search_wikipedia", text.replace("search for", "").strip()),
            "what time": ("tell_time", None),
            "current time": ("tell_time", None),
            "time now": ("tell_time", None),
            "openai": ("ask_openai", text.replace("openai", "").strip()),
            "chatgpt": ("ask_openai", text.replace("chatgpt", "").strip()),
            "exit": ("exit", None),
            "quit": ("exit", None),
            "stop": ("exit", None),
            "date": ("tell_date", None),
            "today's date": ("tell_date", None)
        }
        
        for cmd, (action, detail) in commands.items():
            if cmd in text:
                return action, detail
        
        return "unknown", text

class LLMInterpreter:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = CONFIG["openai_model"]
    
    def interpret_smart(self, text: str):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a command classifier for a voice assistant. Given a user input, classify it into: open_website, search_wikipedia, tell_time, tell_date, ask_openai, exit, or unknown. Extract details. For websites, return full URL. Return JSON: {'command': 'type', 'details': 'value'}"
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=100
            )
            
            raw_reply = response.choices[0].message.content.strip()
            print(f"AI Interpretation: {raw_reply}")
            
            if raw_reply.startswith("```json"):
                raw_reply = raw_reply[7:-3].strip()
            elif raw_reply.startswith("```"):
                raw_reply = raw_reply[3:-3].strip()
            
            result = json.loads(raw_reply)
            return result.get("command", "unknown"), result.get("details")
            
        except json.JSONDecodeError:
            print("Failed to parse AI response as JSON")
            return "unknown", text
        except Exception as e:
            print(f"AI Interpreter error: {e}")
            return "unknown", text

class ActionExecutor:
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.engine = self._init_tts()
    
    def _init_tts(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            voices = engine.getProperty('voices')
            if len(voices) > 1:
                engine.setProperty('voice', voices[1].id)
            return engine
        except Exception as e:
            print(f"TTS initialization failed: {e}")
            return None
    
    def speak(self, text: str):
        print(f"Assistant: {text}")
        if self.engine:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Speech error: {e}")
    
    def get_confirmation(self, command_description: str) -> bool:
        self.speak(f"Did you say: {command_description}?")
        
        try:
            with sr.Microphone() as source:
                print("Waiting for confirmation (say yes/no)...")
                recognizer = sr.Recognizer()
                audio = recognizer.listen(
                    source, 
                    timeout=CONFIG["confirmation_timeout"], 
                    phrase_time_limit=3
                )
                response = recognizer.recognize_google(audio).lower()
                print(f"You said: {response}")
                
                if response in ["yes", "yeah", "yep", "correct", "confirm", "sure", "okay"]:
                    self.speak("Proceeding...")
                    return True
                elif response in ["no", "nope", "cancel", "stop", "wrong", "incorrect"]:
                    self.speak("Cancelled.")
                    return False
                else:
                    self.speak("Please say yes or no clearly.")
                    return self.get_confirmation(command_description)
                    
        except (sr.UnknownValueError, sr.WaitTimeoutError):
            self.speak("I didn't hear a response. Assuming no.")
            return False
        except Exception as e:
            print(f"Confirmation error: {e}")
            return False
    
    def execute(self, command: str, details: str):
        if not self.get_confirmation(f"{command.replace('_', ' ')}: {details}"):
            return
        
        if command == "open_website":
            webbrowser.open(details)
            self.speak(f"Opening {details.split('//')[-1].split('/')[0]}")
        
        elif command == "search_wikipedia":
            try:
                summary = wikipedia.summary(details, sentences=CONFIG["wikipedia_sentences"])
                self.speak(f"According to Wikipedia: {summary}")
            except wikipedia.exceptions.DisambiguationError:
                self.speak(f"Multiple results for {details}. Please be specific.")
            except wikipedia.exceptions.PageError:
                self.speak(f"No Wikipedia page found for {details}.")
            except Exception:
                self.speak("Wikipedia search failed. Check your connection.")
        
        elif command == "tell_time":
            now = datetime.datetime.now().strftime("%I:%M %p")
            self.speak(f"The time is {now}")
        
        elif command == "tell_date":
            today = datetime.datetime.now().strftime("%B %d, %Y")
            self.speak(f"Today is {today}")
        
        elif command == "ask_openai":
            if not self.client:
                self.speak("OpenAI is not configured.")
                return
            try:
                response = self.client.chat.completions.create(
                    model=CONFIG["openai_model"],
                    messages=[
                        {"role": "system", "content": "You are a helpful voice assistant. Keep responses concise (2-3 sentences)."},
                        {"role": "user", "content": details},
                    ],
                    max_tokens=150
                )
                reply = response.choices[0].message.content.strip()
                self.speak(reply)
            except Exception as e:
                self.speak(f"OpenAI error: {str(e)[:50]}...")
        
        elif command == "exit":
            self.speak("Goodbye! Have a great day.")
            sys.exit()
        
        else:
            self.speak("I don't understand that command. Try 'help' for options.")

def main():
    print("="*60)
    print("VOICE ASSISTANT v1.0")
    print("="*60)
    
    api_key = get_api_key()
    speech_input = SpeechInput()
    interpreter = LLMInterpreter(api_key=api_key)
    executor = ActionExecutor(api_key=api_key)
    
    executor.speak("Voice assistant initialized. How can I help you?")
    
    try:
        while True:
            text = speech_input.listen_and_recognize()
            if text:
                print(f"\nCommand: {text}")
                command, details = interpreter.interpret_smart(text)
                executor.execute(command, details)
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("Program terminated")
        executor.speak("Shutting down. Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()