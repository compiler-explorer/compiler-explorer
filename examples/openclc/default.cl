kernel void do_add_sub(global short4 *add_out, global short4 *sub_out,
                       global short4 *x, global short4 *y)
{
    size_t g = get_global_id(0);
    add_out[g] = x[g] + y[g];
    sub_out[g] = x[g] - y[g];
}