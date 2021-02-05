# ----------------------------------- #
class Switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args:  # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False


# ----------------------------------- #

# error code:
# ----------------------------------- #
CODE_ERROR_DATA_UNIT_1 = 0


# ----------------------------------- #


# ----------------------------------- #
def Foo(code):
    for case in Switch(code):
        if case(0):
            print('code error in data_unit: datas size is different with block_width and block_height')

# ----------------------------------- #
