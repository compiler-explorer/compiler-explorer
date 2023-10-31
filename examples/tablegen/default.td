class Register<int _size, string _alias=""> {
    int size = _size;
    string alias = _alias;
}

def X0: Register<8> {}

def X29: Register<8, "frame pointer"> {}