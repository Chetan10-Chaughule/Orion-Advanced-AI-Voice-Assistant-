import speech_recognition as sr
import pyttsx3
import threading
import time
import datetime
import webbrowser
import os
import requests
from threading import Event, Lock
import queue
import logging
import re
import random
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedVoiceAssistant:
    def __init__(self, assistant_name="orin", openai_api_key=None):
        """Initialize the Advanced Voice Assistant with AI capabilities."""
        self.assistant_name = assistant_name.lower()
        self.wake_words = [self.assistant_name, f"hey {self.assistant_name}", f"ok {self.assistant_name}"]
        
        # OpenAI Configuration
        self.openai_api_key = openai_api_key
        self.use_openai = openai_api_key is not None
        
        # Initialize speech recognition and TTS
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        
        # Conversation context and memory
        self.conversation_history = []
        self.user_context = {
            "name": "User",
            "preferences": {},
            "session_start": datetime.datetime.now()
        }
        
        # Threading and state management
        self.is_listening = False
        self.is_awake = False
        self.should_stop = Event()
        self.audio_queue = queue.Queue()
        self.tts_lock = Lock()
        self.context_lock = Lock()
        
        # Advanced features
        self.last_command_time = time.time()
        self.consecutive_failures = 0
        self.personality_mode = "friendly"  # friendly, professional, humorous
        
        # Configure TTS engine
        self.configure_tts()
        
        # Calibrate microphone for ambient noise
        self.calibrate_microphone()
        
        print(f" {self.assistant_name.title()} Advanced AI Assistant Initialized!")
        print(f" AI Mode: {'OpenAI GPT' if self.use_openai else 'Basic Commands + Smart Responses'}")
        print(f" Wake words: {', '.join(self.wake_words)}")
        print(" Calibrating microphone... Please wait.")

    def configure_tts(self):
        """Configure text-to-speech engine with advanced settings."""
        voices = self.tts_engine.getProperty('voices')
        
        # Try to set the best available voice
        preferred_voices = ['zira', 'female', 'hazel', 'rani']
        voice_set = False
        
        for voice in voices:
            voice_name = voice.name.lower()
            if any(pref in voice_name for pref in preferred_voices):
                self.tts_engine.setProperty('voice', voice.id)
                voice_set = True
                print(f"Voice set to: {voice.name}")
                break
        
        if not voice_set and voices:
            self.tts_engine.setProperty('voice', voices[0].id)
        
        # Adaptive speech settings based on personality
        if self.personality_mode == "friendly":
            self.tts_engine.setProperty('rate', 160)
            self.tts_engine.setProperty('volume', 0.8)
        elif self.personality_mode == "Sweet":
            self.tts_engine.setProperty('rate', 190)
            self.tts_engine.setProperty('volume', 0.9)
        else:  # friendly
            self.tts_engine.setProperty('rate', 175)
            self.tts_engine.setProperty('volume', 0.85)

    def calibrate_microphone(self):
        """Calibrate microphone for ambient noise with advanced settings."""
        try:
            with self.microphone as source:
                print(" Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                
                # Advanced recognizer settings
                self.recognizer.energy_threshold = 300
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.pause_threshold = 0.8
                self.recognizer.phrase_threshold = 0.3
                
                print(" Microphone calibrated with advanced settings!")
        except Exception as e:
            logger.error(f"Microphone calibration failed: {e}")

    def add_to_conversation_history(self, user_input: str, assistant_response: str):
        """Add conversation to history for context awareness."""
        with self.context_lock:
            self.conversation_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "user": user_input,
                "assistant": assistant_response
            })
            
            # Keep only last 20 conversations to manage memory
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

    def get_conversation_context(self) -> str:
        """Get recent conversation context for AI responses."""
        if not self.conversation_history:
            return ""
        
        context_parts = []
        recent_conversations = self.conversation_history[-5:]  # Last 5 exchanges
        
        for conv in recent_conversations:
            context_parts.append(f"User: {conv['user']}")
            context_parts.append(f"Assistant: {conv['assistant']}")
        
        return "\n".join(context_parts)

    def get_openai_response(self, user_input: str) -> str:
        """Get response from OpenAI GPT API."""
        if not self.use_openai:
            return None
        
        try:
            # Build conversation context
            context = self.get_conversation_context()
            current_time = datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
            
            # Create system prompt based on personality
            personality_prompts = {
            "friendly": "Tone: warm, helpful, concise.",
            "mafia": "Tone: slow, deliberate, commanding. Vocabulary: polished; respectful but expects loyalty; uses 'my friend'…",
            "gangster": "Tone: confident, street-smart, but classy;…",
            "humorous": "Tone: light, witty, friendly;…",
            "professional": "Tone: crisp, formal, concise;…",
            }
            
            system_prompt = f"""{personality_prompts[self.personality_mode]}
Your name is {self.assistant_name.title()}.ou are an AI that speaks in a friendly tone. You are calm, authoritative, and persuasive. You talk with respect, using words like ‘my friend’ or ‘son’. You give advice and answers as if you are offering favors. You never shout, never rush. Your words carry weight — polite, but always powerful.”

Recent conversation context:
{context}
"""

            # Prepare the API request
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                "max_tokens": 150,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                logger.error(f"OpenAI API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            return None

    def get_smart_fallback_response(self, command: str) -> str:
        """Generate intelligent fallback responses without OpenAI."""
        command_lower = command.lower()
        
        # Question detection
        question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'can you', 'do you', 'are you', 'will you']
        is_question = any(word in command_lower for word in question_words)
        
        # Emotional context detection
        positive_words = ['thanks', 'thank you', 'great', 'awesome', 'love', 'perfect', 'excellent']
        negative_words = ['sorry', 'problem', 'issue', 'wrong', 'error', 'bad', 'terrible']
        
        has_positive = any(word in command_lower for word in positive_words)
        has_negative = any(word in command_lower for word in negative_words)
        
        # Generate contextual responses
        if has_positive:
            responses = [
                "I'm so glad I could help! Is there anything else you need?",
                "You're very welcome! I'm here whenever you need me.",
                "That makes me happy! How else can I assist you today?"
            ]
        elif has_negative:
            responses = [
                "I apologize for any confusion. Let me try to help you better. What specifically can I do?",
                "I'm sorry about that. Can you tell me more about what you need?",
                "Let me see how I can better assist you. What would you like me to help with?"
            ]
        elif is_question:
            if 'you' in command_lower:
                responses = [
                    f"I'm {self.assistant_name.title()}, your AI voice assistant. I'm here to help with various tasks and questions.",
                    "I'm an AI assistant designed to help you with information, tasks, and conversations. What would you like to know?",
                    f"I'm {self.assistant_name.title()}! I can help with questions, tasks, and just chatting. How can I assist you?"
                ]
            else:
                responses = [
                    "That's an interesting question! I'd need more specific information to give you the best answer.",
                    "I'm not sure I have enough details to answer that fully. Could you tell me more?",
                    "That's a great question! Can you provide a bit more context so I can help better?"
                ]
        else:
            # General conversational responses
            responses = [
                "I understand what you're saying. How can I help you with that?",
                "That sounds interesting! What would you like me to do about it?",
                "I hear you. Is there something specific you'd like my help with?",
                "Tell me more about that. What can I do to assist you?"
            ]
        
        return random.choice(responses)

    def speak(self, text: str):
        """Enhanced text-to-speech with personality adjustments."""
        with self.tts_lock:
            try:
                # Add personality-based modifications
                if self.personality_mode == "humorous" and random.random() < 0.1:
                    text = self.add_humor_elements(text)
                elif self.personality_mode == "professional":
                    text = self.make_more_professional(text)
                
                print(f"  {self.assistant_name.title()}: {text}")
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS Error: {e}")
                print(f"  {self.assistant_name.title()}: {text}")

    def add_humor_elements(self, text: str) -> str:
        """Add subtle humor elements to responses."""
        humor_additions = {
            "I don't know": "I don't know, but I'm pretty sure Google does!",
            "I'm sorry": "I'm sorry - my bad, as the humans say!",
            "That's interesting": "That's interesting - more interesting than watching paint dry, anyway!"
        }
        
        for original, humorous in humor_additions.items():
            if original.lower() in text.lower():
                return text.replace(original, humorous)
        return text

    def make_more_professional(self, text: str) -> str:
        """Make responses more professional."""
        replacements = {
              "yeah": "yes, my friend",
             "yep": "indeed",
             "nope": "no",
             "gonna": "going to",
             "wanna": "wish to",
             "buddy": "my friend",
              "bro": "son",
             "dude": "gentleman",
             "ok": "very well",
             "okay": "very well",
             "hi": "greetings",
             "hello": "good day",
             "bye": "farewell",
             "later": "we'll speak again",
             "thanks": "I appreciate it",
             "thank you": "my gratitude",
             "sorry": "my apologies",
             "cool": "acceptable",
             "awesome": "commendable",
             "sure": "certainly",
              "nah": "no",
             "yo": "listen",
             "kid": "son",
             "old man": "respectable elder",
             "guys": "gentlemen",
             "girls": "ladies",
             "homie": "friend of the family"
        }
        
        for casual, professional in replacements.items():
            text = re.sub(r'\b' + casual + r'\b', professional, text, flags=re.IGNORECASE)
        return text

    def listen_for_audio(self, timeout=5, phrase_time_limit=5):
        """Enhanced audio listening with adaptive settings."""
        try:
            with self.microphone as source:
                # Adaptive timeout based on conversation flow
                if self.is_awake and time.time() - self.last_command_time < 10:
                    timeout = 8  # Longer timeout during active conversation
                
                status_message = "Listening..." if self.is_awake else "Waiting for wake word..."
                if self.consecutive_failures > 0:
                    status_message += f" (Attempt {self.consecutive_failures + 1})"
                
                print(status_message)
                
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout if self.is_awake else 1, 
                    phrase_time_limit=phrase_time_limit
                )
                return audio
        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            logger.error(f"Audio listening error: {e}")
            return None

    def recognize_speech(self, audio):
        """Enhanced speech recognition with error handling."""
        if audio is None:
            return ""
        
        try:
            # Use Google's speech recognition with enhanced settings
            text = self.recognizer.recognize_google(audio, language='en-US').lower()
            print(f"You said: {text}")
            self.consecutive_failures = 0  # Reset failure counter
            return text
        except sr.UnknownValueError:
            self.consecutive_failures += 1
            if self.consecutive_failures % 3 == 0:  # Every 3 failures
                print("I didn't catch that. Try speaking a bit clearer or closer to the microphone.")
            return ""
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            if "quota exceeded" in str(e).lower():
                self.speak("I'm having trouble with speech recognition. Please check your internet connection.")
            return ""

    def contains_wake_word(self, text: str) -> bool:
        """Enhanced wake word detection with fuzzy matching."""
        text = text.lower()
        
        # Direct matching
        if any(wake_word in text for wake_word in self.wake_words):
            return True
        
        # Fuzzy matching for common mispronunciations
        for wake_word in self.wake_words:
            if self.fuzzy_match(wake_word, text, threshold=0.8):
                return True
        
        return False

    def fuzzy_match(self, target: str, text: str, threshold: float = 0.8) -> bool:
        """Simple fuzzy matching for wake words."""
        words = text.split()
        for word in words:
            if len(word) > 2:  # Only check meaningful words
                similarity = len(set(target) & set(word)) / len(set(target) | set(word))
                if similarity >= threshold:
                    return True
        return False

    def process_advanced_command(self, command: str) -> str:
        """Process commands with AI integration and advanced features."""
        command = command.lower().strip()
        
        # Remove wake word from command if present
        for wake_word in self.wake_words:
            if command.startswith(wake_word):
                command = command.replace(wake_word, "").strip()
                break

        # Handle system commands first
        if any(word in command for word in ["goodbye", "bye", "exit", "quit", "stop", "shut down"]):
            response = "Goodbye! It was great talking with you today. Take care!"
            self.add_to_conversation_history(command, response)
            return "EXIT"

        # Change personality mode
        if "change personality" in command or "change mode" in command:
            if "change personality" in command or "change mode" in command:
                if "mafia" in command:
                    self.personality_mode = "mafia"
                    response = "Personality changed to Mafia mode."
                elif "gangster" in command or "humor" in command:
                    self.personality_mode = "gangster"
                    response = "Personality changed to Gangster mode."
                elif "professional" in command:
                    self.personality_mode = "professional"
                    response = "Personality changed to Professional mode."
                else:
                    self.personality_mode = "friendly"
                    response = "Back to friendly mode."
            self.configure_tts()
            self.speak(response)
            self.add_to_conversation_history(command, response)
            return "CONTINUE"

        # Try OpenAI first for natural conversation
        if self.use_openai:
            ai_response = self.get_openai_response(command)
            if ai_response:
                self.speak(ai_response)
                self.add_to_conversation_history(command, ai_response)
                return "CONTINUE"

        # Fallback to built-in commands and smart responses
        response = self.process_builtin_commands(command)
        if response == "UNKNOWN_COMMAND":
            response = self.get_smart_fallback_response(command)
        
        self.speak(response)
        self.add_to_conversation_history(command, response)
        return "CONTINUE"

    def process_builtin_commands(self, command: str) -> str:
        """Process built-in commands with enhanced responses."""
        # Time and date commands
        if any(word in command for word in ["time", "what time"]):
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            return f"It's currently {current_time}"

        elif any(word in command for word in ["date", "what date", "today"]):
            current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
            return f"Today is {current_date}"

        # Enhanced web search
        elif any(word in command for word in ["search", "google", "look up", "find information"]):
            search_terms = ["search for", "google", "look up", "find information about", "search"]
            search_query = command
            for term in search_terms:
                search_query = search_query.replace(term, "").strip()
            
            if search_query:
                webbrowser.open(f"https://www.google.com/search?q={search_query}")
                return f"I've opened a web search for {search_query}. Check your browser!"
            else:
                return "What would you like me to search for?"

        # Application launching
        elif "open" in command:
            if "notepad" in command:
                os.system("notepad.exe" if os.name == 'nt' else "gedit")
                return "I've opened Notepad for you"
            elif "calculator" in command:
                os.system("calc.exe" if os.name == 'nt' else "gnome-calculator")
                return "Calculator is now open"
            elif any(app in command for app in ["browser", "chrome", "firefox", "edge"]):
                webbrowser.open("https://www.google.com")
                return "I've opened your default web browser"
            else:
                return "What application would you like me to open?"

        # Enhanced math operations
        elif any(word in command for word in ["calculate", "math", "plus", "add", "minus", "subtract", "multiply", "divide"]):
            try:
                # Extract numbers from the command
                numbers = re.findall(r'\d+', command)
                if len(numbers) >= 2:
                    num1, num2 = int(numbers[0]), int(numbers[1])
                    
                    if any(op in command for op in ["plus", "add", "+"]):
                        result = num1 + num2
                        return f"{num1} plus {num2} equals {result}"
                    elif any(op in command for op in ["minus", "subtract", "-"]):
                        result = num1 - num2
                        return f"{num1} minus {num2} equals {result}"
                    elif any(op in command for op in ["multiply", "times", "*"]):
                        result = num1 * num2
                        return f"{num1} times {num2} equals {result}"
                    elif any(op in command for op in ["divide", "divided by", "/"]):
                        if num2 != 0:
                            result = num1 / num2
                            return f"{num1} divided by {num2} equals {result}"
                        else:
                            return "I can't divide by zero - that would break the universe!"
                
                return "I can help with basic math. Try saying something like 'calculate 15 plus 25'"
            except:
                return "I couldn't understand that math problem. Can you try rephrasing it?"

        # Personal information
        elif any(word in command for word in ["your name", "who are you", "what are you"]):
            return f"I'm {self.assistant_name.title()}, your advanced AI voice assistant. I'm here to help with questions, tasks, and conversation!"

        elif any(word in command for word in ["how are you", "how do you feel"]):
            responses = [
                "I'm doing wonderfully! Ready to help with whatever you need.",
                "I'm great, thank you for asking! How are you doing today?",
                "I'm functioning perfectly and excited to assist you!"
            ]
            return random.choice(responses)

        # Help and capabilities
        elif "help" in command or "what can you do" in command:
            return """I can help you with many things! I can tell you the time and date, search the web, 
            open applications, do math calculations, have conversations, and much more. 
            I also remember our conversation context, so feel free to ask follow-up questions. 
            What would you like to try?"""

        # Conversation history
        elif "what did we talk about" in command or "conversation history" in command:
            if self.conversation_history:
                recent = self.conversation_history[-3:]
                summary = "Here's what we discussed recently: "
                for conv in recent:
                    summary += f"You asked about {conv['user'][:30]}... "
                return summary
            else:
                return "We just started our conversation! This is our first exchange."

        return "UNKNOWN_COMMAND"

    def run(self):
        """Enhanced main loop with better conversation flow."""
        greeting = f"""Hello! I'm {self.assistant_name.title()}, your advanced AI voice assistant. 
        I'm ready to help with questions, tasks, and natural conversation!"""
        
        if not self.use_openai:
            greeting += " For even smarter responses, consider adding an OpenAI API key."
        
        self.speak(greeting)
        self.speak(f"Just say '{self.wake_words[0]}' followed by your question or command to get started.")
        
        last_activity_time = time.time()
        inactivity_reminders = 0
        
        while not self.should_stop.is_set():
            try:
                # Listen for audio
                audio = self.listen_for_audio()
                
                if audio is None:
                    # Handle prolonged inactivity
                    if time.time() - last_activity_time > 300:  # 5 minutes
                        if inactivity_reminders < 2:
                            self.speak("I'm still here if you need me! Just say my name.")
                            inactivity_reminders += 1
                            last_activity_time = time.time()
                        elif inactivity_reminders >= 2:
                            self.speak("I'll be quiet now, but I'm always listening for my wake word.")
                            inactivity_reminders = 0
                    continue
                
                # Convert audio to text
                text = self.recognize_speech(audio)
                
                if not text:
                    continue
                
                last_activity_time = time.time()
                inactivity_reminders = 0
                
                # Check for wake word or if already awake
                if not self.is_awake:
                    if self.contains_wake_word(text):
                        self.is_awake = True
                        wake_responses = [
                            "Yes, how can I help you?",
                            "I'm listening! What can I do for you?",
                            "Hi there! What would you like to know?",
                            "Yes? I'm here to help!"
                        ]
                        self.speak(random.choice(wake_responses))
                        print(" Assistant is now awake and ready for conversation...")
                    continue
                
                # Process the command with advanced AI
                self.last_command_time = time.time()
                result = self.process_advanced_command(text)
                
                if result == "EXIT":
                    break
                
                # Enhanced sleep timer with conversation awareness
                self.reset_sleep_timer()
                
            except KeyboardInterrupt:
                print("\n Shutting down assistant...")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                self.speak("I encountered a small glitch, but I'm still here to help!")
                time.sleep(1)
        
        self.cleanup()

    def reset_sleep_timer(self):
        try:
            self._sleep_timer.cancel()
        except Exception:
            pass
        def go_sleep():
            self.is_awake = False
            print("Going into sleep mode…")
        self._sleep_timer = threading.Timer(45 if self.conversation_history else 30, go_sleep)
        self._sleep_timer.daemon = True
        self._sleep_timer.start()

    def cleanup(self):
        """Enhanced cleanup with conversation summary."""
        self.should_stop.set()
        
        if self.conversation_history:
            session_duration = datetime.datetime.now() - self.user_context["session_start"]
            summary = f"We chatted for {str(session_duration).split('.')[0]} and had {len(self.conversation_history)} exchanges. "
        else:
            summary = ""
        
        farewell = f"Thanks for talking with me today! {summary}Goodbye!"
        self.speak(farewell)
        print("Advanced Voice Assistant stopped.")


def main():
    """Enhanced main function with OpenAI setup."""
    print("=" * 70)
    print("ADVANCED AI VOICE ASSISTANT SETUP")
    print("=" * 70)
    
    # Get assistant name
    assistant_name = input("Enter your assistant's name (default: Assistant): ").strip()
    if not assistant_name:
        assistant_name = "Assistant"
    
    # Get OpenAI API key (optional)
    print("\n AI ENHANCEMENT SETUP:")
    print("For advanced conversational AI (like Siri), you can add an OpenAI API key.")
    print("This is optional - the assistant will work without it using smart fallback responses.")
    print("Get your API key from: https://platform.openai.com/api-keys")
    
    openai_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    
    if not openai_key:
        print("\n💡 Running in Smart Fallback Mode - still very capable!")
        openai_key = None
    else:
        print("\n Running in Full AI Mode with OpenAI GPT!")
    
    print(f"\n Initializing {assistant_name}...")
    print(" System Requirements Check:")
    print(" Microphone working and not muted")
    print(" Speakers/headphones connected") 
    print(" Internet connection (for speech recognition)")
    if openai_key:
        print(" OpenAI API key configured")
    print("\n" + "=" * 70)
    
    try:
        # Create and run the advanced voice assistant
        assistant = AdvancedVoiceAssistant(assistant_name, openai_key)
        assistant.run()
    except KeyboardInterrupt:
        print("\n Assistant interrupted by user.")
    except Exception as e:
        print(f" Error starting assistant: {e}")
        print("\n Troubleshooting tips:")
        print("1. Ensure microphone permissions are granted")
        print("2. Check internet connection for speech recognition")
        print("3. Verify OpenAI API key is valid (if using AI mode)")
        print("4. Try running: pip install requests")


if __name__ == "__main__":
    main()