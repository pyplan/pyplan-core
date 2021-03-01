class ws_settings:
    NOTIFY_USERS_ON_ENTER_OR_LEAVE_GROUPS = False

    # Commands
    COMMANDS = {
        JOIN_GROUP: 'join_group',
        LEAVE_GROUP: 'leave_group',
        SEND_TO_GROUP: 'send_to_group'
    }

    # Types by command
    # join group
    TYPES[COMMAND.JOIN_GROUP] = {
        GROUP_JOIN: 'group.join'
    }
    # leave group
    TYPES[COMMAND.LEAVE_GROUP] = {
        GROUP_LEAVE: 'group.leave'
    }
    # send to group (Types name.name will become methods name_name inside the consumer)
    TYPES[COMMAND.SEND_TO_GROUP] = {
        GROUP_MODEL: 'group.model',
        GROUP_NOTIFICATION: 'group.notification',
        GROUP_SCHEDULETASK = 'group.scheduletask'
    }

    # Messages by type
    # group join messages
    MESSAGES[TYPE[COMMAND.JOIN_GROUP].GROUP_JOIN] = {
        ENTER = 'ENTER GROUP',
    }
    # group leave messages
    MESSAGES[TYPE[COMMAND.LEAVE_GROUP].GROUP_LEAVE] = {
        LEAVE = 'LEAVE GROUP',
    }
    # group model messages
    MESSAGES[TYPE[COMMAND.GROUP_NOTIFICATION].GROUP_MODEL] = {
        OPENING_MODEL: 'OPENING MODEL',
        NODE_PROCESSING: 'NODE PROCESSING',
        NODE_DEBUG_INFORMATION: 'NODE DEBUG INFORMATION'

    }
    # group notification messages
    MESSAGES[TYPE[COMMAND.GROUP_NOTIFICATION].GROUP_NOTIFICATION] = {
        STANDARD: 'STANDARD',
        PROGRESS_BAR: 'PROGRESS BAR',
        KILLED_SESSION: 'KILLED_SESSION'
    }


class not_levels:
    # NOTIFICATION LEVELS (USED FOR TYPE group.notification)
    INFO = 'info'
    SUCCESS = 'success'
    WARNING = 'warning'
    ERROR = 'error'
