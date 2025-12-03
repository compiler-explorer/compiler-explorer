object "Square" {
    code {
        {
            let _1 := memoryguard(0x80)
            mstore(64, _1)
            if callvalue() { revert(0, 0) }
            let _2 := datasize("Square_deployed")
            codecopy(_1, dataoffset("Square_deployed"), _2)
            return(_1, _2)
        }
    }
    object "Square_deployed" {
        code {
            {
                let _1 := memoryguard(0x80)
                mstore(64, _1)
                if iszero(lt(calldatasize(), 4))
                {
                    if eq(0xd27b3841, shr(224, calldataload(0)))
                    {
                        if callvalue() { revert(0, 0) }
                        if slt(add(calldatasize(), not(3)), 32) { revert(0, 0) }
                        let value := calldataload(4)
                        let _2 := and(value, 0xffffffff)
                        if iszero(eq(value, _2)) { revert(0, 0) }
                        let product_raw := mul(_2, _2)
                        let product := and(product_raw, 0xffffffff)
                        if iszero(eq(product, product_raw))
                        {
                            mstore(0, shl(224, 0x4e487b71))
                            mstore(4, 0x11)
                            revert(0, 0x24)
                        }
                        mstore(_1, product)
                        return(_1, 32)
                    }
                }
                revert(0, 0)
            }
        }
        data ".metadata" hex"a26469706673582212209b2b1b86ce0e1a75faa800884ba155bd6bc6a6bc71f210370f818535dcfc5ee364736f6c634300081e0033"
    }
}
