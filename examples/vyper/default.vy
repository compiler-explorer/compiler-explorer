stored_number: uint256

@external
def set_number(num: uint256):
    self.stored_number = num

@external
@view
def get_number() -> uint256:
    return self.stored_number

@external
@view
def square(a: uint256) -> uint256:
    return a * a
