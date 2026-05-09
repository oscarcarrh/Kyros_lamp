import speech_recognition as sr
from google import genai


class VoiceGeminiController:
    def __init__(self):
        self.client = genai.Client()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        with self.microphone as source:
            print("Ajustando ruido ambiente...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

    def listen_text(self):
        try:
            with self.microphone as source:
                print("Escuchando comando...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=4)

            text = self.recognizer.recognize_google(audio, language="es-MX")
            print(f"Comando detectado: {text}")
            return text

        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            print("No entendí el audio.")
            return None
        except Exception as e:
            print(f"Error escuchando voz: {e}")
            return None

    def command_to_action(self, text):
        if not text:
            return "NONE"

        prompt = f"""
Eres un controlador de una lámpara inteligente en una simulación.

Tu tarea es convertir el comando del usuario en una sola acción.

Acciones válidas:
- TURN_ON: cuando el usuario quiere encender la lámpara.
- TURN_OFF: cuando el usuario quiere apagar la lámpara.
- NONE: cuando no hay una intención clara.

Responde únicamente con una de estas palabras:
TURN_ON, TURN_OFF o NONE.

Comando del usuario:
{text}
"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )

            action = response.text.strip().upper()
            print(f"Acción Gemini: {action}")

            if action in ["TURN_ON", "TURN_OFF", "NONE"]:
                return action

            return "NONE"

        except Exception as e:
            print(f"Error con Gemini: {e}")
            return "NONE"

    def get_voice_action(self):
        text = self.listen_text()
        return self.command_to_action(text)