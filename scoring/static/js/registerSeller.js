//注册
function doRegister(){
    var account=$("#account").val();
    var pwd=$("#pwd").val();
    var pwdAgain=$("#pwdAgain").val();
    var email=$("#email").val();
    var userName=$("#userName").val();
    var phone=$("#phone").val();
    if(userName==''){
        layer.msg('用户名不能为空');
        return;
    }
    if(account==''){
        layer.msg('账号不能为空');
        return;
    }
     if(pwd==''){
        layer.msg('密码不能为空');
        return;
    }
     if(pwd!=pwdAgain){
        layer.msg('两次密码输入不一致');
        return;
    }
     $.ajax({
        //请求方式
        type : "POST",
        //请求地址
        url : "./doRegisterSeller",
        //数据，json字符串
        data : {account:account,pwd:pwd,email:email,phone:phone,userName:userName},
        //请求成功
        success : function(resstr) {
            res=JSON.parse(resstr)
            if(res.code==200){
                layer.msg('注册成功');
                setTimeout(function(){
                   location.href="./admin/home"
                },1500)
            }else{
                layer.msg('账号已存在，请重试');
            }
        },
        //请求失败，包含具体的错误信息
        error : function(e){
            layer.msg('系统错误，稍后请重试');
        }
    });
}