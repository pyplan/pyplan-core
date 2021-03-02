# Commands
class _COMMANDS(object):
    def __init__(self):
        self.JOIN_GROUP = 'join_group'
        self.LEAVE_GROUP = 'leave_group'
        self.SEND_TO_GROUP = 'send_to_group'

# Types by command
class _TYPES(object):
    def __init__(self, command):
        # join group
        if command == _COMMANDS().JOIN_GROUP:
            self.GROUP_JOIN = 'group.join'
        # leave group
        elif command == _COMMANDS().LEAVE_GROUP:
            self.GROUP_LEAVE = 'group.leave'
        # send to group (Types name.name will become methods name_name inside the consumer)
        elif command == _COMMANDS().SEND_TO_GROUP:
            self.GROUP_MODEL = 'group.model'
            self.GROUP_NOTIFICATION = 'group.notification'
            self.GROUP_SCHEDULETASK = 'group.scheduletask'

# Messages by type
class _MESSAGES(object):
    def __init__(self, _type):
        # group join messages
        if _type == _TYPES(_COMMANDS().JOIN_GROUP).GROUP_JOIN:
            self.ENTER_GROUP = 'ENTER_GROUP'
        # group leave messages
        elif _type == _TYPES(_COMMANDS().LEAVE_GROUP).GROUP_LEAVE:
            self.LEAVE_GROUP = 'LEAVE_GROUP'
        # group model messages
        elif _type == _TYPES(_COMMANDS().SEND_TO_GROUP).GROUP_MODEL:
            self.OPENING_MODEL = 'OPENING_MODEL'
            self.NODE_BUTTON_FINISH_PROCESSING = 'NODE_BUTTON_FINISH_PROCESSING'
            self.NODE_DEBUG_INFORMATION = 'NODE_DEBUG_INFORMATION'
        # group notification messages
        elif _type == _TYPES(_COMMANDS().SEND_TO_GROUP).GROUP_NOTIFICATION:
            self.STANDARD = 'STANDARD'
            self.PROGRESS_BAR = 'PROGRESS_BAR'
            self.KILLED_SESSION = 'KILLED_SESSION'


class ws_settings(object):

    NOTIFY_USERS_ON_ENTER_OR_LEAVE_GROUPS = False

    @classmethod
    def COMMANDS(cls):
        return _COMMANDS()

    @classmethod
    def TYPES(cls, _command):
        return _TYPES(_command)

    @classmethod
    def MESSAGES(cls, _type):
        return _MESSAGES(_type)


class not_levels:
    # NOTIFICATION LEVELS (USED FOR TYPE group.notification)
    INFO = 'info'
    SUCCESS = 'success'
    WARNING = 'warning'
    ERROR = 'error'
