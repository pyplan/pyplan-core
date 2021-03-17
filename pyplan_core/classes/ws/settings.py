from enum import Enum

# PRIVATE HELPERS


class _ValEnum(Enum):
    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

# types enums


class _GroupJoin(_ValEnum):
    GROUP_JOIN = 'group.join'


class _GroupLeave(_ValEnum):
    GROUP_LEAVE = 'group.leave'


class _GroupSend(_ValEnum):
    GROUP_MODEL = 'group.model'
    GROUP_NOTIFICATION = 'group.notification'
    GROUP_SCHEDULETASK = 'group.scheduletask'

# messages enums


class _EnterGroup(_ValEnum):
    ENTER_GROUP = 'ENTER_GROUP'


class _LeaveGroup(_ValEnum):
    LEAVE_GROUP = 'LEAVE_GROUP'


class _ModelGroup(_ValEnum):
    OPENING_MODEL = 'OPENING_MODEL'
    NODE_BUTTON_FINISH_PROCESSING = 'NODE_BUTTON_FINISH_PROCESSING'
    NODE_DEBUG_INFORMATION = 'NODE_DEBUG_INFORMATION'


class _NotificacionGroup(_ValEnum):
    STANDARD = 'STANDARD'
    PROGRESS_BAR = 'PROGRESS_BAR'
    KILLED_SESSION = 'KILLED_SESSION'


# WS COMMANDS, TYPES AND MESSAGES
class Commands(Enum):
    JOIN_GROUP = 'join_group'
    LEAVE_GROUP = 'leave_group'
    SEND_TO_GROUP = 'send_to_group'


class TypesFactory(object):
    def __new__(self, command: Commands):
        if command == Commands.JOIN_GROUP:
            return super(TypesFactory, self).__new__(_GroupJoin)
        elif command == Commands.LEAVE_GROUP:
            return super(TypesFactory, self).__new__(_GroupLeave)
        elif command == Commands.SEND_TO_GROUP:
            return super(TypesFactory, self).__new__(_GroupSend)


class MessagesFactory(object):
    def __new__(self, _type):
        if _type == TypesFactory(Commands.JOIN_GROUP).GROUP_JOIN:
            return super(MessagesFactory, self).__new__(_EnterGroup)
        elif _type == TypesFactory(Commands.LEAVE_GROUP).GROUP_LEAVE:
            return super(MessagesFactory, self).__new__(_LeaveGroup)
        elif _type == TypesFactory(Commands.SEND_TO_GROUP).GROUP_MODEL:
            return super(MessagesFactory, self).__new__(_ModelGroup)
        elif _type == TypesFactory(Commands.SEND_TO_GROUP).GROUP_NOTIFICATION:
            return super(MessagesFactory, self).__new__(_NotificacionGroup)

# NOTIFICATION LEVELS (USED FOR TYPE group.notification)


class NotLevels(Enum):
    INFO = 'info'
    SUCCESS = 'success'
    WARNING = 'warning'
    ERROR = 'error'
