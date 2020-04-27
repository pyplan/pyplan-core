from classes.Model import Model
from multiprocessing import Lock

import os


class Pyplan(object):
    """Main class to interact with the pyplan model"""

    def __init__(self):
        self.model = Model()
        self.lastUpdated = -1
        self.currentFilename = ""
        self.lock = Lock()

    def openModel(self, filename):
        """Open Pyplan model"""
        try:
            self.lock_acquire()
            self.currentFilename = filename
            self.lastUpdated = os.path.getmtime(filename)
            self.model.openModel(filename)
        finally:
            self.lock_release()

    def closeModel(self):
        """Close current model"""
        try:
            self.lock_acquire()
            self.model.closeModel()
        finally:
            self.lock_release()

    def getResult(self, node_id):
        """Return result of the node"""
        try:
            self.lock_acquire()
            return self.model.getNode(node_id).result
        finally:
            self.lock_release()

    def setSelectorValue(self, node_id, value):
        """Set value of a selector node"""
        try:
            self.lock_acquire()
            self.model.setSelectorValue(node_id, value)
        finally:
            self.lock_release()

    def checkForReloadModel(self):
        """Check and reolad model if the last update time has changed"""
        if os.path.getmtime(self.currentFilename) != self.lastUpdated:
            self.openModel(self.currentFilename)

    def lock_acquire(self):
        if not self.lock is None:
            return self.lock.acquire()
        return False

    def lock_release(self):
        if not self.lock is None:
            try:
                self.lock.release()
            except:
                pass
