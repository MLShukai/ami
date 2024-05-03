from ami.threads.thread_control import ThreadController, ThreadControllerStatus
from ami.threads.web_api_handler import WebApiHandler

controller = ThreadController()
status = ThreadControllerStatus(controller)
handler = WebApiHandler(status)
handler.run()
