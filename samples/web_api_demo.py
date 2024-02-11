from ami.threads.thread_control import ThreadController
from ami.threads.web_api_handler import WebApiHandler

controller = ThreadController()
handler = WebApiHandler(controller)
handler.run()
