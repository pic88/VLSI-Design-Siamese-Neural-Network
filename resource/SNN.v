module SNN(
           //Input Port
           clk,
           rst_n,
           in_valid,
           Img,
           Kernel,
           Weight,
           Opt,

           //Output Port
           out_valid,
           out
       );


//---------------------------------------------------------------------
//   PARAMETER
//---------------------------------------------------------------------

// IEEE floating point parameter
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
parameter inst_arch_type = 0;
parameter inst_arch = 0;
parameter inst_faithful_round = 0;

input rst_n, clk, in_valid;
input [inst_sig_width+inst_exp_width:0] Img, Kernel, Weight;
input [1:0] Opt;

output reg	out_valid;
output reg [inst_sig_width+inst_exp_width:0] out;

// fsm parameter
parameter WAIT_INPUT = 0;
parameter WAIT_STORE_DATA = 1;
parameter PADDING = 8;
parameter CONVOLUTION = 2;
parameter MAX_POOLING = 3;
parameter FULL_CONNECT_AND_FLATTEN = 4;
parameter NORMALIZATION = 5;
parameter ACTIVE_FUNCTION = 6;
parameter DISTANCE = 7;

// fsm's trigger signal
reg store_data_trigger;
reg start_padding_trigger;
reg start_convulution_trigger;
reg start_max_pooling_trigger;
reg start_full_connect_trigger;
reg start_normalization_trigger;
reg start_active_fuc_trigger;
reg start_distance_trigger;
reg out_ans;
// fsm_state
reg [3:0] ps,ns;

// reg for data
reg [inst_sig_width + inst_exp_width : 0]  img [0:5][0:3][0:3];
reg [inst_sig_width + inst_exp_width : 0]  kernel [0:2][0:2][0:2];
reg [inst_sig_width + inst_exp_width : 0]  weight [0:1][0:1];
reg [1:0] opt;

// reg for img
reg [3:0] img_i,img_j,img_k;
reg start_store_img;
reg finish_store_img;

// reg for img
reg [3:0] kernel_i,kernel_j,kernel_k;
reg start_store_kernel;
reg finish_store_kernel;

//reg for weight
reg [3:0] weight_i,weight_j;
reg start_store_weight;
reg finish_store_weight;


//reg for padding
reg [inst_sig_width + inst_exp_width : 0]  padding_img [0:5][0:5][0:5];
integer padding_i,padding_j,padding_k;

//reg wire for conv
reg [2:0] conv_i,conv_j,conv_k;
reg [2:0] conv_k_mod,conv_k_div;
reg start_conv,finish_conv;
wire [inst_sig_width + inst_exp_width : 0] conv_ans;
reg [inst_sig_width + inst_exp_width : 0] conv_input [0:8];
reg start_con_module;
wire con_out_valid;

//reg for featuremap
reg [inst_sig_width + inst_exp_width : 0] feature_map [0:1][0:3][0:3];

integer fi,fj,fk;
wire [inst_sig_width + inst_exp_width : 0] feature_tmp_ans;

//reg for max_pooling
reg [3:0] mp_i,mp_j,mp_k;
reg strart_max_pooling;
reg finish_max_pooling;
reg [inst_sig_width + inst_exp_width : 0] max_pooling_rst [0:1][0:1][0:1];
wire [inst_sig_width + inst_exp_width : 0] max_pooling_tmp_ans;
wire [inst_sig_width + inst_exp_width : 0] max_pooling_tmp_ans2;

//reg for full_connect
reg [inst_sig_width + inst_exp_width : 0]  full_connect_result [0:1][0:3];
reg [2:0] fct_i,fct_j,fct_k,fct_l;
reg  start_fct,wait_fct,finish_fct;
wire [inst_sig_width + inst_exp_width : 0]  fct_tmp_ans1,fct_tmp_ans2,fct_tmp_ans;
reg [inst_sig_width + inst_exp_width : 0] fct_ans1_reg,fct_ans2_reg;
//reg for  Normalization

reg [inst_sig_width + inst_exp_width : 0]  normal_result [0:1][0:3];
reg normal_i;
reg [3:0] normal_j;
wire [inst_sig_width + inst_exp_width : 0] normal_min,normal_max;
reg start_normalize,finish_normalize;
wire [inst_sig_width + inst_exp_width : 0] up,down,scaled;

//reg for active function
wire [inst_sig_width + inst_exp_width : 0] exp_z,exp_nz,exp_deno;
wire [inst_sig_width + inst_exp_width : 0] ez_mins_ezn;
reg exp_i;
reg [3:0] exp_j;
reg start_exp,finish_exp;
wire [inst_sig_width + inst_exp_width : 0] exp_ans;
wire [inst_sig_width + inst_exp_width : 0] down_left,upper;
reg [inst_sig_width + inst_exp_width : 0]  exp_result [0:1][0:3];
reg  [inst_sig_width + inst_exp_width : 0] exp_z_reg,exp_nz_reg;
reg  [inst_sig_width + inst_exp_width : 0] exp_deno_reg,ez_mins_ezn_reg;
reg caculate_z,sub_z;

//reg for distance

reg [3:0] dis_i;
wire [inst_sig_width + inst_exp_width : 0] dis_mins,dis_tmp_ans;
reg [inst_sig_width + inst_exp_width : 0] dis_ans;
reg start_dis,finish_dis;

//==============================================//
//                  main fsm                    //
//==============================================//

always @(posedge clk or negedge rst_n)
begin
    if(~rst_n)
        ps <= WAIT_INPUT;
    else
        ps <= ns;
end

always @(*)
begin
    ns = ps;
    store_data_trigger = 0;
    start_padding_trigger = 0;
    start_convulution_trigger = 0;
    start_max_pooling_trigger = 0;
    start_full_connect_trigger = 0;
    start_normalization_trigger = 0;
    start_active_fuc_trigger = 0;
    start_distance_trigger = 0;
    out_ans = 0;
    case (ps)
        WAIT_INPUT:
        begin
            if(in_valid)
            begin
                store_data_trigger = 1;
                ns = WAIT_STORE_DATA;
            end
        end
        WAIT_STORE_DATA:
        begin
            if(finish_store_img && finish_store_kernel && finish_store_weight)
            begin
                ns = PADDING;
                start_padding_trigger = 1;
            end
        end
        PADDING:
        begin
            ns = CONVOLUTION;
            start_convulution_trigger = 1;
        end
        CONVOLUTION:
        begin
            if(finish_conv)
            begin
                ns = MAX_POOLING;
                start_max_pooling_trigger = 1;
            end
        end
        MAX_POOLING:
        begin
            if(finish_max_pooling)
            begin
                start_full_connect_trigger = 1;
                ns = FULL_CONNECT_AND_FLATTEN;
            end
        end
        FULL_CONNECT_AND_FLATTEN:
        begin
            if(finish_fct)
            begin
                start_normalization_trigger = 1;
                ns = NORMALIZATION;
            end
        end
        NORMALIZATION:
        begin
            if(finish_normalize)
            begin
                start_active_fuc_trigger = 1;
                ns = ACTIVE_FUNCTION;
            end
        end
        ACTIVE_FUNCTION:
        begin
            if(finish_exp)
            begin
                start_distance_trigger = 1;
                ns = DISTANCE;
            end
        end
        DISTANCE:
        begin
            if(finish_dis)
            begin
                ns = WAIT_INPUT;
                out_ans = 1;
            end
        end
        default:
            ns = WAIT_INPUT;
    endcase
end



//==============================================//
//                  output                      //
//==============================================//
always @(posedge clk or negedge rst_n)
begin
    if(~rst_n)
    begin
        out <= 0;
        out_valid <= 0;
    end
    else if (out_ans)
    begin
        out <= dis_ans;
        out_valid <= 1;
    end
    else
    begin
        out <= 0;
        out_valid <= 0;
    end
end

//==============================================//
//                  STORE DATA                  //
//==============================================//
/* store img*/
always @(posedge clk or negedge rst_n)
begin
    if(~rst_n)
    begin
        img_i <= 0;
        img_j <= 0;
        img_k <= 0;
        start_store_img <= 0;
        finish_store_img <= 0;
    end
    else if(store_data_trigger)
    begin
        img_i <= 0;
        img_j <= 1;
        img_k <= 0;
        start_store_img <= 1;
        finish_store_img <= 0;
        img[img_k][img_i][img_j] <= Img;
    end
    else if(start_store_img)
    begin
        img[img_k][img_i][img_j] <= Img;
        if(img_i == 3 && img_j== 3 && img_k == 5)
        begin
            img_k <= img_k;
            img_i <= img_i;
            img_j <= img_j;
            start_store_img <= 0;
            finish_store_img <= 1;
        end
        else if(img_i == 3 && img_j== 3)
        begin
            img_k <= img_k + 1;
            img_i <= 0;
            img_j <= 0;
            start_store_img <= 1;
            finish_store_img <= 0;
        end
        else if(img_j == 3)
        begin
            img_k <= img_k;
            img_i <= img_i + 1;
            img_j <= 0;
            start_store_img <= 1;
            finish_store_img <= 0;
        end
        else
        begin
            img_k <= img_k;
            img_i <= img_i;
            img_j <= img_j + 1;
            start_store_img <= 1;
            finish_store_img <= 0;
        end
    end
    else
    begin
        img_i <= 0;
        img_j <= 0;
        img_k <= 0;
        start_store_img <= 0;
        finish_store_img <= finish_store_img;
    end
end

/* store kernel*/
always @(posedge clk or negedge rst_n)
begin
    if(~rst_n)
    begin
        kernel_i <= 0;
        kernel_j <= 0;
        kernel_k <= 0;
        start_store_kernel <= 0;
        finish_store_kernel <= 0;
    end
    else if(store_data_trigger)
    begin
        kernel_i <= 0;
        kernel_j <= 1;
        kernel_k <= 0;
        start_store_kernel <= 1;
        finish_store_kernel <= 0;
        kernel[0][0][0] <= Kernel;
    end
    else if(start_store_kernel)
    begin
        kernel[kernel_k][kernel_i][kernel_j] <= Kernel;
        if(kernel_i == 2 && kernel_j== 2 && kernel_k == 2)
        begin
            kernel_k <= kernel_k;
            kernel_i <= kernel_i;
            kernel_j <= kernel_j;
            start_store_kernel <= 0;
            finish_store_kernel <= 1;
        end
        else if(kernel_i == 2 && kernel_j== 2)
        begin
            kernel_k <= kernel_k + 1;
            kernel_i <= 0;
            kernel_j <= 0;
            start_store_kernel <= 1;
            finish_store_kernel <= 0;
        end
        else if(kernel_j == 2)
        begin
            kernel_k <= kernel_k;
            kernel_i <= kernel_i + 1;
            kernel_j <= 0;
            start_store_kernel <= 1;
            finish_store_kernel <= 0;
        end
        else
        begin
            kernel_k <= kernel_k;
            kernel_i <= kernel_i;
            kernel_j <= kernel_j + 1;
            start_store_kernel <= 1;
            finish_store_kernel <= 0;
        end
    end
    else
    begin
        kernel_i <= 0;
        kernel_j <= 0;
        kernel_k <= 0;
        start_store_kernel <= 0;
        finish_store_kernel <= finish_store_kernel;
    end
end

/* store weight*/
always @(posedge clk or negedge rst_n)
begin
    if(~rst_n)
    begin
        weight_i <= 0;
        weight_j <= 0;
        start_store_weight <= 0;
        finish_store_weight <= 0;
    end
    else if(store_data_trigger)
    begin
        weight_i <= 0;
        weight_j <= 1;
        start_store_weight <= 1;
        finish_store_weight <= 0;
        weight[0][0] <= Weight;
    end
    else if(start_store_weight)
    begin
        weight[weight_i][weight_j] <= Weight;
        if(weight_i == 1 && weight_j== 1)
        begin
            weight_i <= weight_i;
            weight_j <= weight_j;
            start_store_weight <= 0;
            finish_store_weight <= 1;
        end
        else if(weight_j == 1)
        begin
            weight_i <= weight_i + 1;
            weight_j <= 0;
            start_store_weight <= 1;
            finish_store_weight <= 0;
        end
        else
        begin
            weight_i <= weight_i;
            weight_j <= weight_j + 1;
            start_store_weight <= 1;
            finish_store_weight <= 0;
        end
    end
    else
    begin
        weight_i <= 0;
        weight_j <= 0;
        start_store_weight <= 0;
        finish_store_weight <= finish_store_weight;
    end
end

/* store opt*/
always @(posedge clk or negedge rst_n)
begin
    if(~rst_n)
        opt <= 0;
    else if(store_data_trigger)
        opt <= Opt;
    else
        opt <= opt;
end

//==============================================//
//                  PADDING                    //
//==============================================//


always @(posedge clk)
begin
    if(start_padding_trigger)
    begin
        for(padding_k = 0; padding_k < 6; padding_k = padding_k + 1)
        begin
            padding_img[padding_k][0][0] <= opt[0] ? 0 : img[padding_k][0][0];
            padding_img[padding_k][0][1] <= opt[0] ? 0 : img[padding_k][0][0];
            padding_img[padding_k][0][2] <= opt[0] ? 0 : img[padding_k][0][1];
            padding_img[padding_k][0][3] <= opt[0] ? 0 : img[padding_k][0][2];
            padding_img[padding_k][0][4] <= opt[0] ? 0 : img[padding_k][0][3];
            padding_img[padding_k][0][5] <= opt[0] ? 0 : img[padding_k][0][3];
            padding_img[padding_k][1][0] <= opt[0] ? 0 : img[padding_k][0][0];
            padding_img[padding_k][1][5] <= opt[0] ? 0 : img[padding_k][0][3];
            padding_img[padding_k][2][0] <= opt[0] ? 0 : img[padding_k][1][0];
            padding_img[padding_k][2][5] <= opt[0] ? 0 : img[padding_k][1][3];
            padding_img[padding_k][3][0] <= opt[0] ? 0 : img[padding_k][2][0];
            padding_img[padding_k][3][5] <= opt[0] ? 0 : img[padding_k][2][3];
            padding_img[padding_k][4][0] <= opt[0] ? 0 : img[padding_k][3][0];
            padding_img[padding_k][4][5] <= opt[0] ? 0 : img[padding_k][3][3];
            padding_img[padding_k][5][0] <= opt[0] ? 0 : img[padding_k][3][0];
            padding_img[padding_k][5][1] <= opt[0] ? 0 : img[padding_k][3][0];
            padding_img[padding_k][5][2] <= opt[0] ? 0 : img[padding_k][3][1];
            padding_img[padding_k][5][3] <= opt[0] ? 0 : img[padding_k][3][2];
            padding_img[padding_k][5][4] <= opt[0] ? 0 : img[padding_k][3][3];
            padding_img[padding_k][5][5] <= opt[0] ? 0 : img[padding_k][3][3];
            for(padding_i = 1;padding_i < 5;padding_i = padding_i + 1)
            begin
                for(padding_j = 1;padding_j <5;padding_j = padding_j + 1)
                begin
                    padding_img[padding_k][padding_i][padding_j] <= img[padding_k][padding_i-1][padding_j-1];
                end
            end
        end
    end
end

//==============================================//
//                  CONVOLUTION                 //
//==============================================//
always @(posedge clk or negedge rst_n)
begin
    if(~rst_n)
    begin
        conv_i <= 0;
        conv_j <= 0;
        conv_k <= 0;
        conv_k_mod <= 0;
        conv_k_div <= 0;
        finish_conv <= 0;
        start_con_module <= 0;
    end
    else if(start_convulution_trigger)
    begin
        conv_i <= 0;
        conv_j <= 0;
        conv_k <= 0;
        conv_k_mod <= 0;
        conv_k_div <= 0;
        finish_conv <= 0;
        start_con_module <= 1;
        for(fk = 0;fk <2 ;fk = fk + 1)
        begin
            for(fi = 0;fi < 4;fi = fi + 1)
            begin
                for(fj = 0;fj < 4;fj = fj + 1)
                begin
                    feature_map[fk][fi][fj] <= 0;
                end
            end
        end
    end
    else if(finish_conv)
    begin
        conv_i <= 0;
        conv_j <= 0;
        conv_k <= 0;
        conv_k_mod <= 0;
        conv_k_div <= 0;
        start_con_module <= 0;
        finish_conv <= finish_conv;
    end
    else if(con_out_valid)
    begin
        finish_conv <= 0;
        start_con_module <= 1;
        feature_map[conv_k_div][conv_i][conv_j] <= feature_tmp_ans;
        if(conv_i == 3 && conv_j==3 && conv_k == 5)
        begin
            start_conv <= 0;
            finish_conv <= 1;
            conv_i <= conv_i;
            conv_j <= conv_j;
            conv_k <= conv_k;
            conv_k_mod <= conv_k_mod;
            conv_k_div <= conv_k_div;
        end
        else if(conv_i == 3 && conv_j==3 && conv_k == 2)
        begin
            start_conv <= 1;
            finish_conv <= 0;
            conv_i <= 0;
            conv_j <= 0;
            conv_k <= conv_k + 1;
            conv_k_mod <= 3'b0;
            conv_k_div <= 1;
        end
        else if(conv_i == 3 && conv_j==3)
        begin
            conv_i <= 0;
            conv_j <= 0;
            conv_k <= conv_k + 1;
            conv_k_mod <= conv_k_mod + 1;
            conv_k_div <= conv_k_div;
        end
        else if(conv_j==3)
        begin
            conv_i <= conv_i + 1;
            conv_j <= 0;
            conv_k <= conv_k;
            conv_k_mod <= conv_k_mod;
            conv_k_div <= conv_k_div;
        end
        else
        begin
            conv_i <= conv_i;
            conv_j <= conv_j + 1;
            conv_k <= conv_k;
            conv_k_mod <= conv_k_mod;
            conv_k_div <= conv_k_div;
        end
    end
    else
    begin
        conv_i <= conv_i;
        conv_j <= conv_j;
        conv_k <= conv_k;
        conv_k_mod <= conv_k_mod;
        conv_k_div <= conv_k_div;
        start_con_module <= 0;
        finish_conv <= 0;
    end
end

DW_fp_addsub#(inst_sig_width,inst_exp_width,inst_ieee_compliance)
            AC (.a(feature_map[conv_k_div][conv_i][conv_j]), .b(conv_ans), .op(1'd0), .rnd(3'd0), .z(feature_tmp_ans));

wire [2:0] conv_j_1,conv_j_2;
wire [2:0] conv_i_1,conv_i_2;
assign conv_i_1 = conv_i + 1;
assign conv_i_2 = conv_i + 2;
assign conv_j_1 = conv_j + 1;
assign conv_j_2 = conv_j + 2;

convulution #(inst_sig_width, inst_exp_width, inst_ieee_compliance)
            cnv(
                .a0(padding_img[conv_k][conv_i][conv_j]),
                .a1(padding_img[conv_k][conv_i][conv_j_1]),
                .a2(padding_img[conv_k][conv_i][conv_j_2]),
                .a3(padding_img[conv_k][conv_i_1][conv_j]),
                .a4(padding_img[conv_k][conv_i_1][conv_j_1]),
                .a5(padding_img[conv_k][conv_i_1][conv_j_2]),
                .a6(padding_img[conv_k][conv_i_2][conv_j]),
                .a7(padding_img[conv_k][conv_i_2][conv_j_1]),
                .a8(padding_img[conv_k][conv_i_2][conv_j_2]),
                .b0(kernel[conv_k_mod][0][0]),
                .b1(kernel[conv_k_mod][0][1]),
                .b2(kernel[conv_k_mod][0][2]),
                .b3(kernel[conv_k_mod][1][0]),
                .b4(kernel[conv_k_mod][1][1]),
                .b5(kernel[conv_k_mod][1][2]),
                .b6(kernel[conv_k_mod][2][0]),
                .b7(kernel[conv_k_mod][2][1]),
                .b8(kernel[conv_k_mod][2][2]),
                .rst_n(rst_n),
                .start_con(start_con_module),
                .clk(clk),
                .out_valid(con_out_valid),
                .out_ans(conv_ans)
            );

//==============================================//
//                 MAX_POOLING                  //
//==============================================//

always @(posedge clk or negedge rst_n)
begin
    if(~rst_n)
    begin
        mp_i <= 0;
        mp_j <= 0;
        mp_k <= 0;
        strart_max_pooling <= 0;
        finish_max_pooling <= 0;
    end
    else if(start_max_pooling_trigger)
    begin
        mp_i <= 0;
        mp_j <= 0;
        mp_k <= 0;
        strart_max_pooling <= 1;
        finish_max_pooling <= 0;
    end
    else if(strart_max_pooling)
    begin
        strart_max_pooling <= 1;
        finish_max_pooling <= 0;
        max_pooling_rst[mp_k][mp_i][mp_j] <= max_pooling_tmp_ans;
        if(mp_i == 1 && mp_j == 1 && mp_k == 1)
        begin
            strart_max_pooling <= 0;
            finish_max_pooling <= 1;
            mp_i <= 0;
            mp_j <= 0;
            mp_k <= 0;
        end
        else if(mp_i == 1 && mp_j == 1)
        begin
            mp_i <= 0;
            mp_j <= 0;
            mp_k <= mp_k + 1;
        end
        else if( mp_j == 1)
        begin
            mp_i <= mp_i + 1;
            mp_j <= 0;
            mp_k <= mp_k;
        end
        else
        begin
            mp_i <= mp_i;
            mp_j <= mp_j + 1;
            mp_k <= mp_k;
        end
    end
    else
    begin
        mp_i <= 0;
        mp_j <= 0;
        mp_k <= 0;
        strart_max_pooling <= 0;
        finish_max_pooling <= finish_max_pooling;
    end
end

max_min #(inst_sig_width,inst_exp_width,inst_ieee_compliance) maxmin(
            feature_map[mp_k][mp_i*2][mp_j*2],feature_map[mp_k][mp_i*2][mp_j*2 + 1],
            feature_map[mp_k][mp_i*2+1][mp_j*2],feature_map[mp_k][mp_i*2+1][mp_j*2 + 1],
            max_pooling_tmp_ans2,max_pooling_tmp_ans
        );

//==============================================//
//                 FULL_CONNECT_AND_FLATTEN     //
//==============================================//

DW_fp_mult#(inst_sig_width, inst_exp_width, inst_ieee_compliance)
          fct (.a(max_pooling_rst[fct_k][fct_i][0]), .b(weight[0][fct_j]), .rnd(3'd0), .z(fct_tmp_ans1));
DW_fp_mult#(inst_sig_width, inst_exp_width, inst_ieee_compliance)
          fct1 (.a(max_pooling_rst[fct_k][fct_i][1]), .b(weight[1][fct_j]), .rnd(3'd0), .z(fct_tmp_ans2));
DW_fp_addsub#(inst_sig_width,inst_exp_width,inst_ieee_compliance)
            A3 (.a(fct_tmp_ans1), .b(fct_tmp_ans2), .op(1'd0), .rnd(3'd0), .z(fct_tmp_ans));

always @(posedge clk or negedge rst_n)
begin
    if(~rst_n)
    begin
        fct_i <= 0;
        fct_j <= 0;
        fct_k <= 0;
        fct_l <= 0;
        start_fct <= 0;
        finish_fct <= 0;
        wait_fct <= 0;
        fct_ans1_reg <= 0;
        fct_ans2_reg <= 0;
    end
    else if(start_full_connect_trigger)
    begin
        fct_i <= 0;
        fct_j <= 0;
        fct_k <= 0;
        fct_l <= 0;
        start_fct <= 1;
        finish_fct <= 0;
        wait_fct <= 0;
        fct_ans1_reg <= 0;
        fct_ans2_reg <= 0;
    end
    else if(start_fct)
    begin
        fct_i <= fct_i;
        fct_j <= fct_j;
        fct_k <= fct_k;
        fct_l <= fct_l;
        wait_fct <= 1;
        start_fct <= 0;
        finish_fct <= 0;
        fct_ans1_reg <= fct_tmp_ans1;
        fct_ans2_reg <= fct_tmp_ans2;
    end
    else if(wait_fct)
    begin
        start_fct <= 1;
        finish_fct <= 0;
        wait_fct <= 0;
        fct_ans1_reg <= fct_ans1_reg;
        fct_ans2_reg <= fct_ans2_reg;
        full_connect_result[fct_k][fct_l] <= fct_tmp_ans;
        if(fct_j == 1 && fct_i == 1 && fct_k == 1)
        begin
            fct_i <= 0;
            fct_j <= 0;
            fct_k <= 0;
            fct_l <= 0;
            start_fct <= 0;
            finish_fct <= 1;
        end
        else if(fct_j == 1 && fct_i == 1)
        begin
            fct_i <= 0;
            fct_j <= 0;
            fct_l <= 0;
            fct_k <= fct_k + 1;
        end
        else if(fct_j == 1)
        begin
            fct_i <= fct_i + 1;
            fct_j <= 0;
            fct_l <= fct_l + 1;
            fct_k <= fct_k;
        end
        else
        begin
            fct_i <= fct_i;
            fct_j <= fct_j + 1;
            fct_l <= fct_l + 1;
            fct_k <= fct_k;
        end
    end
    else
    begin
        fct_i <= 0;
        fct_j <= 0;
        fct_k <= 0;
        fct_l <= 0;
        start_fct <= 0;
        wait_fct <= 0;
        finish_fct <= finish_fct;
        fct_ans1_reg <= fct_ans1_reg;
        fct_ans2_reg <= fct_ans2_reg;
    end
end

//==============================================//
//                 Min-Max Normalization        //
//==============================================//
max_min #(inst_sig_width,inst_exp_width,inst_ieee_compliance) maxmin_normalize(
            full_connect_result[normal_i][0],full_connect_result[normal_i][1],
            full_connect_result[normal_i][2],full_connect_result[normal_i][3],
            normal_min,normal_max
        );

DW_fp_sub
    #(inst_sig_width,inst_exp_width,inst_ieee_compliance)
    normalsub1 (.a(normal_max), .b(normal_min), .z(down), .rnd(3'd0));

DW_fp_sub
    #(inst_sig_width,inst_exp_width,inst_ieee_compliance)
    normalsub2 (.a(full_connect_result[normal_i][normal_j]), .b(normal_min), .z(up), .rnd(3'd0));

DW_fp_div // 1 / [1+exp(-x)]
          #(inst_sig_width,inst_exp_width,inst_ieee_compliance, 0)
          normadiv (.a(up), .b(down), .rnd(3'd0), .z(scaled));


always @(posedge clk or negedge rst_n)
begin
    if(~rst_n)
    begin
        normal_i <= 0;
        normal_j <= 0;
        start_normalize <= 0;
        finish_normalize <= 0;
    end
    else if(start_normalization_trigger)
    begin
        normal_i <= 0;
        normal_j <= 0;
        start_normalize <= 1;
        finish_normalize <= 0;
    end
    else if(start_normalize)
    begin
        normal_result[normal_i][normal_j] <= scaled;
        start_normalize <= 1;
        finish_normalize <= 0;
        if(normal_j==3 && normal_i == 1)
        begin
            normal_i <= 0;
            normal_j <= 0;
            start_normalize <= 0;
            finish_normalize <= 1;
        end
        else if(normal_j==3)
        begin
            normal_i <= normal_i + 1;
            normal_j <= 0;
        end
        else
        begin
            normal_j <= normal_j + 1;
            normal_i <= normal_i;
        end
    end
    else
    begin
        normal_i <= 0;
        normal_j <= 0;
        start_normalize <= 0;
        finish_normalize <= finish_normalize;
    end
end

//==============================================//
//             Activation Function              //
//==============================================//

assign down_left =  opt[1] ? exp_z_reg : 32'h3F800000;
assign upper = opt[1] ? ez_mins_ezn_reg :  32'h3F800000;
DW_fp_exp // exp(-x)
          #(inst_sig_width,inst_exp_width,inst_ieee_compliance, inst_arch)
          EXPNZ (.a({~normal_result[exp_i][exp_j][31],normal_result[exp_i][exp_j][30:0]}), .z(exp_nz));

DW_fp_exp // exp(x)
          #(inst_sig_width,inst_exp_width,inst_ieee_compliance, inst_arch)
          EXPZ (.a(normal_result[exp_i][exp_j]), .z(exp_z));

DW_fp_addsub // deno
             #(inst_sig_width,inst_exp_width,inst_ieee_compliance)
             exp_dneo (.a(down_left), .b(exp_nz_reg), .op(1'd0), .rnd(3'd0), .z(exp_deno));

DW_fp_sub
    #(inst_sig_width,inst_exp_width,inst_ieee_compliance)
    ez_mins_ezns (.a(exp_z_reg), .b(exp_nz_reg), .z(ez_mins_ezn), .rnd(3'd0));


DW_fp_div // exp ans
          #(inst_sig_width,inst_exp_width,inst_ieee_compliance, 0)
          exp_div1 (.a(upper), .b(exp_deno_reg), .rnd(3'd0), .z(exp_ans));

always @(posedge clk or negedge rst_n)
begin
    if(~rst_n)
    begin
        // i & j
        exp_i <= 0;
        exp_j <= 0;

        start_exp <= 0;
        finish_exp <= 0;
        caculate_z <= 0;
        sub_z <= 0;

        //reg continue
        exp_z_reg <= 0;
        exp_nz_reg <= 0;
        exp_deno_reg <= 0;
        ez_mins_ezn_reg <= 0;

    end
    else if(start_active_fuc_trigger)
    begin
        exp_i <= 0;
        exp_j <= 0;

        //fsm triger
        start_exp <= 0;
        finish_exp <= 0;
        caculate_z <= 1;
        sub_z <= 0;

        exp_z_reg <= 0;
        exp_nz_reg <= 0;
        exp_deno_reg <= 0;
        ez_mins_ezn_reg <= 0;

    end
    else if(caculate_z)
    begin
        exp_i <= exp_i;
        exp_j <= exp_j;

        //fsm triger
        start_exp <= 0;
        finish_exp <= 0;
        caculate_z <= 0;
        sub_z <= 1;


        exp_z_reg <= exp_z;
        exp_nz_reg <= exp_nz;
        exp_deno_reg <= exp_deno_reg;
        ez_mins_ezn_reg <= ez_mins_ezn_reg;
    end
    else if(sub_z)
    begin
        exp_i <= exp_i;
        exp_j <= exp_j;

        start_exp <= 1;
        finish_exp <= 0;
        caculate_z <= 0;
        sub_z <= 0;

        exp_z_reg <= exp_z_reg;
        exp_nz_reg <= exp_nz_reg;
        exp_deno_reg <= exp_deno;
        ez_mins_ezn_reg <= ez_mins_ezn;
    end
    else if(start_exp)
    begin

        exp_z_reg <= exp_z_reg;
        exp_nz_reg <= exp_nz_reg;
        exp_deno_reg <= exp_deno_reg;
        ez_mins_ezn_reg <= ez_mins_ezn_reg;

        exp_result[exp_i][exp_j] <= exp_ans;
        start_exp <= 0;
        finish_exp <= 0;
        caculate_z <= 1;
        sub_z <= 0;

        if(exp_j==3 && exp_i == 1)
        begin
            exp_i <= 0;
            exp_j <= 0;
            finish_exp <= 1;
        end
        else if(exp_j==3)
        begin
            exp_i <= exp_i + 1;
            exp_j <= 0;
        end
        else
        begin
            exp_j <= exp_j + 1;
            exp_i <= exp_i;
        end
    end
    else
    begin
        exp_i <= 0;
        exp_j <= 0;

        caculate_z <= 0;
        sub_z <= 0;
        start_exp <= 0;
        finish_exp <= finish_exp;


        exp_z_reg <= 0;
        exp_nz_reg <= 0;
        exp_deno_reg <= 0;
        ez_mins_ezn_reg <= 0;
    end

end


//==============================================//
//             DISTANCE                         //
//==============================================//

DW_fp_sub
    #(inst_sig_width,inst_exp_width,inst_ieee_compliance)
    dissub (.a(exp_result[0][dis_i]), .b(exp_result[1][dis_i]), .z(dis_mins), .rnd(3'd0));

DW_fp_addsub // dneo
             #(inst_sig_width,inst_exp_width,inst_ieee_compliance)
             dis_add(.a({1'b0,dis_mins[30:0]}), .b(dis_ans), .op(1'd0), .rnd(3'd0), .z(dis_tmp_ans));

always @(posedge clk or negedge rst_n)
begin
    if(~rst_n)
    begin
        dis_i <= 0;
        dis_ans <= 0;
        start_dis <= 0;
        finish_dis <= 0;
    end
    else if(start_distance_trigger)
    begin
        dis_i <= 0;
        dis_ans <= 0;
        start_dis <= 1;
        finish_dis <= 0;
    end
    else if(start_dis)
    begin
        dis_ans <= dis_tmp_ans;
        start_dis <= 1;
        finish_dis <= 0;
        if(dis_i == 3)
        begin
            dis_i <= 0;
            start_dis <= 0;
            finish_dis <= 1;
        end
        else
        begin
            dis_i <= dis_i + 1;
        end
    end
    else
    begin
        dis_i <= 0;
        dis_ans <= dis_ans;
        start_dis <= 0;
        finish_dis <= finish_dis;
    end
end

endmodule


    module convulution
    #(  parameter inst_sig_width       = 23,
        parameter inst_exp_width       = 8,
        parameter inst_ieee_compliance = 1
     )
    (
        input  [inst_sig_width+inst_exp_width:0] a0, a1, a2, a3, a4, a5, a6, a7, a8,
        input  [inst_sig_width+inst_exp_width:0] b0, b1, b2, b3, b4, b5, b6, b7, b8,
        input rst_n,
        input start_con,
        input clk,
        output reg out_valid,
        output reg [inst_sig_width+inst_exp_width:0] out_ans
    );

wire [inst_sig_width+inst_exp_width:0] pixel0, pixel1, pixel2, pixel3, pixel4, pixel5, pixel6, pixel7, pixel8;
reg [inst_sig_width+inst_exp_width:0] pixel0_reg, pixel1_reg, pixel2_reg, pixel3_reg, pixel4_reg, pixel5_reg, pixel6_reg, pixel7_reg, pixel8_reg;
reg start_mult,plus0,plus1,plus2,plus3;
wire [inst_sig_width+inst_exp_width:0] out;
wire [inst_sig_width+inst_exp_width:0] add0, add1, add2, add3, add4, add5, add6;
reg [inst_sig_width+inst_exp_width:0] add0_reg, add1_reg, add2_reg, add3_reg, add4_reg, add5_reg, add6_reg;

always @(posedge clk or negedge rst_n)
begin
    if(~rst_n)
    begin
        plus0 <= 0;
        pixel0_reg <= 0;
        pixel1_reg <= 0;
        pixel2_reg <= 0;
        pixel3_reg <= 0;
        pixel4_reg <= 0;
        pixel5_reg <= 0;
        pixel6_reg <= 0;
        pixel7_reg <= 0;
        pixel8_reg <= 0;
    end
    else if(start_con)
    begin
        plus0 <= 1;
        pixel0_reg <= pixel0;
        pixel1_reg <= pixel1;
        pixel2_reg <= pixel2;
        pixel3_reg <= pixel3;
        pixel4_reg <= pixel4;
        pixel5_reg <= pixel5;
        pixel6_reg <= pixel6;
        pixel7_reg <= pixel7;
        pixel8_reg <= pixel8;
    end
    else
    begin
        plus0 <= 0;
        pixel0_reg <= pixel0_reg;
        pixel1_reg <= pixel1_reg;
        pixel2_reg <= pixel2_reg;
        pixel3_reg <= pixel3_reg;
        pixel4_reg <= pixel4_reg;
        pixel5_reg <= pixel5_reg;
        pixel6_reg <= pixel6_reg;
        pixel7_reg <= pixel7_reg;
        pixel8_reg <= pixel8_reg;
    end
end

always @(posedge clk or negedge rst_n)
begin
    if(~rst_n)
    begin
        plus1 <= 0;
        add0_reg <= 0;
        add1_reg <= 0;
        add2_reg <= 0;
        add3_reg <= 0;
    end
    else if(plus0)
    begin
        plus1 <= 1;
        add0_reg <= add0;
        add1_reg <= add1;
        add2_reg <= add2;
        add3_reg <= add3;
    end
    else
    begin
        plus1 <= 0;
        add0_reg <= add0_reg;
        add1_reg <= add1_reg;
        add2_reg <= add2_reg;
        add3_reg <= add3_reg;
    end
end

always @(posedge clk or negedge rst_n)
begin
    if(~rst_n)
    begin
        plus2 <= 0;
        add4_reg <= 0;
        add5_reg <= 0;
    end
    else if(plus1)
    begin
        plus2 <= 1;
        add4_reg <= add4;
        add5_reg <= add5;
    end
    else
    begin
        plus2 <= 0;
        add4_reg <= add4_reg;
        add5_reg <= add5_reg;
    end
end

always @(posedge clk or negedge rst_n)
begin
    if(~rst_n)
    begin
        plus3 <= 0;
        add6_reg <= 0;
    end
    else if(plus2)
    begin
        plus3 <= 1;
        add6_reg <= add6;
    end
    else
    begin
        plus3 <= 0;
        add6_reg <= add6_reg;
    end
end

always @(posedge clk or negedge rst_n)
begin
    if(~rst_n)
    begin
        out_valid <= 0;
        out_ans <= 0;
    end
    else if(plus3)
    begin
        out_valid <= 1;
        out_ans <= out;
    end
    else
    begin
        out_valid <= 0;
        out_ans <= out_ans;
    end
end

// Multiplication
DW_fp_mult#(inst_sig_width, inst_exp_width, inst_ieee_compliance)
          M0 (.a(a0), .b(b0), .rnd(3'd0), .z(pixel0));

DW_fp_mult#(inst_sig_width, inst_exp_width, inst_ieee_compliance)
          M1 (.a(a1), .b(b1), .rnd(3'd0), .z(pixel1));

DW_fp_mult#(inst_sig_width, inst_exp_width, inst_ieee_compliance)
          M2 (.a(a2), .b(b2), .rnd(3'd0), .z(pixel2));

DW_fp_mult#(inst_sig_width, inst_exp_width, inst_ieee_compliance)
          M3 (.a(a3), .b(b3), .rnd(3'd0), .z(pixel3));

DW_fp_mult#(inst_sig_width, inst_exp_width, inst_ieee_compliance)
          M4 (.a(a4), .b(b4), .rnd(3'd0), .z(pixel4));

DW_fp_mult#(inst_sig_width, inst_exp_width, inst_ieee_compliance)
          M5 (.a(a5), .b(b5), .rnd(3'd0), .z(pixel5));

DW_fp_mult#(inst_sig_width, inst_exp_width, inst_ieee_compliance)
          M6 (.a(a6), .b(b6), .rnd(3'd0), .z(pixel6));

DW_fp_mult#(inst_sig_width, inst_exp_width, inst_ieee_compliance)
          M7 (.a(a7), .b(b7), .rnd(3'd0), .z(pixel7));

DW_fp_mult#(inst_sig_width, inst_exp_width, inst_ieee_compliance)
          M8 (.a(a8), .b(b8), .rnd(3'd0), .z(pixel8));


// Addition plus0
DW_fp_addsub#(inst_sig_width,inst_exp_width,inst_ieee_compliance)
            A0 (.a(pixel0_reg), .b(pixel1_reg), .op(1'd0), .rnd(3'd0), .z(add0));

DW_fp_addsub#(inst_sig_width,inst_exp_width,inst_ieee_compliance)
            A1 (.a(pixel2_reg), .b(pixel3_reg), .op(1'd0), .rnd(3'd0), .z(add1));

DW_fp_addsub#(inst_sig_width,inst_exp_width,inst_ieee_compliance)
            A2 (.a(pixel4_reg), .b(pixel5_reg), .op(1'd0), .rnd(3'd0), .z(add2));

DW_fp_addsub#(inst_sig_width,inst_exp_width,inst_ieee_compliance)
            A3 (.a(pixel6_reg), .b(pixel7_reg), .op(1'd0), .rnd(3'd0), .z(add3));


// plus1
DW_fp_addsub#(inst_sig_width,inst_exp_width,inst_ieee_compliance)
            A4 (.a(pixel8_reg), .b(add0_reg), .op(1'd0), .rnd(3'd0), .z(add4));

DW_fp_addsub#(inst_sig_width,inst_exp_width,inst_ieee_compliance)
            A5 (.a(add1_reg), .b(add2_reg), .op(1'd0), .rnd(3'd0), .z(add5));

//plus2
DW_fp_addsub#(inst_sig_width,inst_exp_width,inst_ieee_compliance)
            A6 (.a(add3_reg), .b(add4_reg), .op(1'd0), .rnd(3'd0), .z(add6));

DW_fp_addsub#(inst_sig_width,inst_exp_width,inst_ieee_compliance)
            A7 (.a(add5_reg), .b(add6_reg), .op(1'd0), .rnd(3'd0), .z(out));
endmodule


    module max_min
    #(  parameter inst_sig_width       = 23,
        parameter inst_exp_width       = 8,
        parameter inst_ieee_compliance = 1
     )
    (
        input  [inst_sig_width+inst_exp_width:0] a0, a1, a2, a3,
        output [inst_sig_width+inst_exp_width:0] min, max
    );

wire [inst_sig_width+inst_exp_width:0] max0,max1,min0,min1;
wire f1, f2,f3,f4;

DW_fp_cmp
    #(inst_sig_width,inst_exp_width,inst_ieee_compliance)
    C0_1 (.a(a0), .b(a1), .agtb(f1), .zctr(1'd0));
DW_fp_cmp
    #(inst_sig_width,inst_exp_width,inst_ieee_compliance)
    C1_2 (.a(a2), .b(a3), .agtb(f2), .zctr(1'd0));
assign max0 = f1 ? a0 : a1;
assign max1 = f2 ? a2 : a3;

DW_fp_cmp
    #(inst_sig_width,inst_exp_width,inst_ieee_compliance)
    Cmax (.a(max0), .b(max1), .agtb(f3), .zctr(1'd0));
DW_fp_cmp
    #(inst_sig_width,inst_exp_width,inst_ieee_compliance)
    Cmin (.a(min0), .b(min1), .agtb(f4), .zctr(1'd0));

assign min0 = f1 ? a1 : a0;
assign min1 = f2 ? a3 : a2;
assign max = f3 ? max0 : max1;
assign min = f4 ? min1 : min0;

endmodule



