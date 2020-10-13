#include <array>

namespace NA {

    template<class Derived, typename DataType>
    class Iterator {
    public:
        Iterator() = default;

        class iterator {
        public:
            iterator(DataType* ptr = nullptr) : ptr(ptr) {}
            inline iterator operator++() {
                ++ptr;
                return *this;
            }
            inline bool operator!=(const iterator& other) const { return ptr != other.ptr; }
            inline DataType& operator*() { return *ptr; }

        private:
            DataType* ptr;
        };

        class const_iterator {
        public:
            const_iterator(const DataType* ptr = nullptr) : ptr(ptr) {}
            inline const_iterator operator++() {
                ++ptr;
                return *this;
            }
            inline bool operator!=(const iterator& other) const { return ptr != other.ptr; }
            inline const DataType& operator*() const { return *ptr; }

        private:
            const DataType* ptr;
        };

        inline iterator begin() { return iterator((DataType*)((Derived*)this)->IteratorBegin()); }
        inline iterator end() { return iterator((DataType*)((Derived*)this)->IteratorEnd()); }
        inline const_iterator begin() const { return const_iterator((const DataType*)((Derived*)this)->IteratorBegin()); }
        inline const_iterator end() const { return const_iterator((const DataType*)((Derived*)this)->IteratorEnd()); }
    };

} // namespace NA


namespace NA {
    template<typename T, unsigned int MAX_SIZE>
    class FixedStack : public Iterator<FixedStack<T, MAX_SIZE>, T> {
        friend class Iterator<FixedStack<T, MAX_SIZE>, T>;
        inline constexpr auto IteratorBegin() { return m_Stack.data(); }
        inline constexpr auto IteratorEnd() { return m_Stack.data() + m_Pos + 1; }

    public:
        FixedStack() : m_Pos(-1) {}

        inline bool Push(T elem) {
            if ((m_Pos + 1) < MAX_SIZE) {
                m_Pos++;
                m_Stack[m_Pos] = elem;
                return true;
            }

            return false;
        }

        inline bool Pop(T replaceLast) {
            if (m_Pos != -1) {
                m_Stack[m_Pos--] = replaceLast;
                return true;
            }

            return false;
        }

        inline bool Pop() {
            if (m_Pos != -1) {
                m_Pos--;
                return true;
            }

            return false;
        }

        inline void Clear(T fill) {
            m_Pos = -1;
            m_Stack.fill(fill);
        }

        inline void Clear() { m_Pos = -1; }

        inline int FreeSpace() { return (MAX_SIZE - (m_Pos + 1)); }

        inline T* DataPtr() const { return m_Stack.data(); }

        inline int Size() const { return m_Pos + 1; }

        inline bool Full() { return Size() == MAX_SIZE; }

        inline T& Top(int back = 0) {
             return m_Stack[m_Pos + back];
        }

        inline bool Empty() const { return m_Pos == -1; }

        inline T& operator[](int index) { return m_Stack[index]; }
        inline const T& operator[](int index) const { return m_Stack[index]; }

    private:
        int m_Pos;
        std::array<T, MAX_SIZE> m_Stack;
    };

} // namespace NA

NA::FixedStack<int, 16> desa;

void main() {


    desa.Push(1);
    desa.Push(2);
    desa.Push(3);
    desa.Push(4);

    for(auto d : desa) {
        printf("%d\n", d);
    }
}